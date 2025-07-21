import os
import sys
import time
import logging
import importlib.util
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from mmaction.apis import init_recognizer
from mmaction.models.backbones.swin import PatchEmbed3D
from tqdm import tqdm

# Patch PatchEmbed3D 支持4D输入（必要时才用）
_orig_pe_forward = PatchEmbed3D.forward
def _patched_pe_forward(self, x):
    if x.ndim == 4:
        BD, D, H, W = x.shape
        C = self.in_channels
        B = BD // C
        x = x.view(B, C, D, H, W)
    return _orig_pe_forward(self, x)
PatchEmbed3D.forward = _patched_pe_forward

def linear_rampup(current_epoch, rampup_length):
    if rampup_length == 0:
        return 1.0
    return float(np.clip(current_epoch / rampup_length, 0.0, 1.0))

def setup_logger(log_dir, log_name="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, log_name), encoding='utf-8')
    fh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def load_cfg_py(cfg_path):
    spec = importlib.util.spec_from_file_location("cfg", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["cfg"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

def extract_semi_params(cfg):
    p = {}
    p['config']        = cfg.__file__
    p['checkpoint']    = cfg.model['backbone'].get('pretrained', None)
    p['train_csv']     = cfg.ann_file_train.replace('.txt', '.csv')
    val = getattr(cfg, 'ann_file_val', None)
    p['val_csv']       = val.replace('.txt', '.csv') if val and val.endswith('.txt') else (val or p['train_csv'].replace('train', 'val'))
    p['unlabeled_csv'] = cfg.ann_file_unlabeled
    p['rawframes_dir'] = cfg.data_root
    p['ddpm_npy_dir']  = getattr(cfg, 'ddpm_logits_dir', None)
    assert p['ddpm_npy_dir'], "请在 cfg 中定义 ddpm_logits_dir"
    sample = cfg.train_dataloader['dataset']['pipeline'][0]
    p['clip_len']      = sample.get('clip_len', 8)
    p['bs']            = cfg.train_dataloader.get('batch_size', 8)
    p['lr']            = cfg.optim_wrapper['optimizer'].get('lr', 5e-3)
    p['lambda_u']      = getattr(cfg, 'lambda_u', 1.0)
    p['rampup_epochs'] = getattr(cfg, 'rampup_epochs', 5)
    p['epochs']        = cfg.train_cfg.get('max_epochs', 30)
    p['save_dir']      = cfg.work_dir if hasattr(cfg, 'work_dir') else 'work_dirs/student'
    return p

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, raw_dir, clip_len, transform=None):
        self.df = pd.read_csv(csv_file, header=None, names=['pid','frame','label'])
        self.df = self.df[self.df['label'].astype(str).str.isdigit()]
        self.df = self.df.drop_duplicates(['pid','label'])
        self.raw_dir, self.clip_len, self.transform = raw_dir, clip_len, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        pid = str(self.df.iloc[idx]['pid'])
        label = int(self.df.iloc[idx]['label'])
        from PIL import Image
        imgs = [Image.open(os.path.join(self.raw_dir, pid, f'img_{i:04d}.jpg')).convert('RGB')
                for i in range(1, self.clip_len+1)]
        if self.transform:
            imgs = [self.transform(im) for im in imgs]
        video = torch.stack(imgs).permute(1,0,2,3)
        return video, label

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, raw_dir, clip_len, npy_dir, transform=None):
        with open(txt_file) as f:
            self.pids = [l.strip() for l in f if l.strip()]
        self.raw_dir, self.clip_len = raw_dir, clip_len
        self.npy_dir, self.transform = npy_dir, transform
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, idx):
        pid = self.pids[idx]
        from PIL import Image
        imgs = [Image.open(os.path.join(self.raw_dir, pid, f'img_{i:04d}.jpg')).convert('RGB')
                for i in range(1, self.clip_len+1)]
        if self.transform:
            imgs = [self.transform(im) for im in imgs]
        video = torch.stack(imgs).permute(1,0,2,3)
        feat = np.load(os.path.join(self.npy_dir, f"{pid}.npy"))
        feat = torch.from_numpy(feat).float()
        return video, feat

def evaluate_on_loader(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            features = model.extract_feat(x)
            if isinstance(features, (tuple, list)):
                features = features[0]
            logits = model.cls_head(features)
            loss = crit(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
    return correct/total, loss_sum/total

# 投影头
class FeatProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.fc(x)

def train_student(params):
    logger = setup_logger(params['save_dir'])
    writer = SummaryWriter(os.path.join(params['save_dir'], 'tb'))
    logger.info(f"Starting training, params:\n{params}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = init_recognizer(params['config'], params['checkpoint'], device=device)

    # ==== 初始化投影头 ====
    example = np.load(os.path.join(params['ddpm_npy_dir'], os.listdir(params['ddpm_npy_dir'])[0]))
    print("DDPM feature shape:", example.shape)
    feat_dim = np.prod(example.shape)
    num_classes = 2  # 按你实际类别修改
    proj_head = FeatProjector(feat_dim, num_classes).to(device)

    sup_crit  = nn.CrossEntropyLoss()
    kl_crit   = nn.KLDivLoss(reduction='batchmean')
    T         = 2.0
    rampup_e  = params['rampup_epochs']

    optimizer = optim.SGD(
        list(student.parameters()) + list(proj_head.parameters()),
        lr=params['lr'], momentum=0.9, weight_decay=1e-4
    )

    lab_loader = DataLoader(
        LabeledDataset(params['train_csv'], params['rawframes_dir'],
                       params['clip_len'], transform=get_transform()),
        batch_size=params['bs'], shuffle=True, drop_last=True, num_workers=4
    )
    unlab_loader = DataLoader(
        UnlabeledDataset(params['unlabeled_csv'], params['rawframes_dir'],
                         params['clip_len'], params['ddpm_npy_dir'], transform=get_transform()),
        batch_size=params['bs'], shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = DataLoader(
        LabeledDataset(params['val_csv'], params['rawframes_dir'],
                       params['clip_len'], transform=get_transform()),
        batch_size=params['bs'], shuffle=False, num_workers=4
    )

    best_val_acc = 0.0
    for epoch in range(1, params['epochs']+1):
        t0 = time.time()
        student.train()
        proj_head.train()
        un_iter = iter(unlab_loader)
        sup_sum, cons_sum = 0.0, 0.0

        with tqdm(total=len(lab_loader), desc=f"Epoch {epoch}/{params['epochs']}") as pbar:
            for x_l, y_l in lab_loader:
                x_l, y_l = x_l.to(device), y_l.to(device)
                # 无标签批次
                try:
                    x_u, feat_u = next(un_iter)
                except StopIteration:
                    un_iter = iter(unlab_loader)
                    x_u, feat_u = next(un_iter)
                x_u, feat_u = x_u.to(device), feat_u.to(device)
                feat_u = feat_u.view(feat_u.size(0), -1)

                # ---- supervised ----
                features_l = student.extract_feat(x_l)
                if isinstance(features_l, (tuple, list)):
                    features_l = features_l[0]
                logits_l = student.cls_head(features_l)
                l_sup = sup_crit(logits_l, y_l)

                # ---- DDPM 蒸馏 ----
                features_u = student.extract_feat(x_u)
                if isinstance(features_u, (tuple, list)):
                    features_u = features_u[0]
                s_logits_u = student.cls_head(features_u)
                proj_feat = proj_head(feat_u)

                # 检查 shape 是否正确
                if (s_logits_u.shape[1] != num_classes or proj_feat.shape[1] != num_classes):
                    print(f"Shape mismatch: s_logits_u {s_logits_u.shape}, proj_feat {proj_feat.shape}")
                    raise RuntimeError("Shape mismatch in student/proj_head output!")

                # Softmax 蒸馏
                p_s = torch.log_softmax(s_logits_u / T, dim=1)
                p_t = torch.softmax(proj_feat / T, dim=1)
                l_cons = kl_crit(p_s, p_t) * (T * T)

                # ramp-up 权重
                λ_u = params['lambda_u'] * linear_rampup(epoch, rampup_e)

                loss = l_sup + λ_u * l_cons
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sup_sum  += l_sup.item()
                cons_sum += l_cons.item()
                step = (epoch-1) * len(lab_loader) + pbar.n + 1
                writer.add_scalar('Loss/Sup',  l_sup.item(),  step)
                writer.add_scalar('Loss/Cons', l_cons.item(), step)

                pbar.set_postfix({'sup':f"{l_sup:.3f}", 'cons':f"{l_cons:.3f}", 'λ_u':f"{λ_u:.3f}"})
                pbar.update()

        # —— 评估 & 日志 ——#
        t_elapsed = time.time() - t0
        m, s = divmod(t_elapsed, 60)
        train_acc, _ = evaluate_on_loader(student, lab_loader, device)
        val_acc, val_loss = evaluate_on_loader(student, val_loader, device)
        logger.info(
            f"[Epoch {epoch:2d}] sup={sup_sum/len(lab_loader):.4f} | "
            f"cons={cons_sum/len(lab_loader):.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | time={int(m)}m{int(s)}s"
        )
        writer.add_scalar('Eval/ValAcc', val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            fp = os.path.join(params['save_dir'], f"best_student_epoch_{epoch}.pth")
            torch.save(student.state_dict(), fp)
            logger.info(f"*** Saved best at epoch {epoch}, val_acc={val_acc:.4f} ***")

    # 保存最终
    torch.save(student.state_dict(), os.path.join(params['save_dir'], "student_final.pth"))
    logger.info("Training finished.")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python xxx.py <config_py>"
    cfg_path = sys.argv[1]
    cfg = load_cfg_py(cfg_path)
    params = extract_semi_params(cfg)
    base = os.path.splitext(os.path.basename(cfg_path))[0]
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    params['save_dir'] = os.path.join(params['save_dir'], f"{base}_{ts}")
    train_student(params)


    #12213