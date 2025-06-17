# fix_checkpoint_swin.py
from mmaction.models.backbones.swin import PatchEmbed3D
_PatchEmbed3D_forward = PatchEmbed3D.forward
def _patched_forward(self, x):
    if x.ndim == 4:
        BD, D, H, W = x.shape
        C = self.in_channels
        B = BD // C
        x = x.view(B, C, D, H, W)
    return _PatchEmbed3D_forward(self, x)
PatchEmbed3D.forward = _patched_forward

import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from mmaction.apis import init_recognizer

# ============ 日志与归档功能 ============
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

def setup_logger(log_dir, log_name="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def save_config(cfg_path, save_dir, params):
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(save_dir, "config.py"))
    with open(os.path.join(save_dir, "run_params.txt"), "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def load_cfg_py(cfg_path):
    spec = importlib.util.spec_from_file_location("cfg", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["cfg"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

def extract_semi_params(cfg):
    params = dict()
    params['config'] = cfg.__file__
    params['checkpoint'] = cfg.model['backbone'].get('pretrained', None)
    params['train_csv'] = cfg.ann_file_train.replace('.txt', '.csv')
    params['unlabeled_csv'] = getattr(cfg, 'ann_file_unlabeled', None)
    params['rawframes_dir'] = cfg.data_root
    params['clip_len'] = 8
    params['bs'] = cfg.train_dataloader.get('batch_size', 8)
    params['unlab_mult'] = 1
    params['lr'] = cfg.optim_wrapper['optimizer'].get('lr', 0.005)
    params['tau'] = 0.95
    params['lambda_u'] = 1.0
    params['epochs'] = cfg.train_cfg.get('max_epochs', 30)
    params['save_dir'] = cfg.work_dir if hasattr(cfg, 'work_dir') else 'work_dirs/teacher_student'
    params['ema_decay'] = 0.99
    params['val_csv'] = getattr(cfg, 'ann_file_val', params['train_csv'].replace('train', 'val'))
    return params

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, rawframes_dir, clip_len=8, transform=None):
        self.df = pd.read_csv(file_path, sep=None, engine='python', header=None, names=['patch_id', 'frame_num', 'label'])
        self.df = self.df[self.df['label'] != 'label']
        self.df = self.df[self.df['label'].astype(str).str.strip() != '']
        self.df = self.df[~self.df['label'].isnull()]
        self.df = self.df[self.df['label'].apply(lambda x: str(x).isdigit())]
        self.df = self.df.drop_duplicates(subset=['patch_id', 'label'])
        self.rawframes_dir = rawframes_dir
        self.clip_len = clip_len
        self.transform = transform
        if len(self.df) == 0:
            raise ValueError(f"没有可用的有效数据，请检查 {file_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = str(row['patch_id'])
        label = int(row['label'])
        video = self.load_frames(patch_id)
        if self.transform:
            video = [self.transform(img) for img in video]
        video = torch.stack(video)               # [T, 3, H, W]
        video = video.permute(1, 0, 2, 3)        # [3, T, H, W]
        return video, label

    def load_frames(self, patch_id):
        from PIL import Image
        patch_dir = os.path.join(self.rawframes_dir, patch_id)
        imgs = []
        for i in range(1, self.clip_len+1):
            img_path = os.path.join(patch_dir, f'img_{i:04d}.jpg')
            img = Image.open(img_path).convert('RGB')
            imgs.append(img)
        return imgs

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, rawframes_dir, clip_len=8, transform=None):
        with open(txt_file) as f:
            self.patch_ids = [line.strip() for line in f if line.strip()]
        self.rawframes_dir = rawframes_dir
        self.clip_len = clip_len
        self.transform = transform

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        video = self.load_frames(patch_id)
        if self.transform:
            video = [self.transform(img) for img in video]
        video = torch.stack(video)
        video = video.permute(1, 0, 2, 3)
        return video

    def load_frames(self, patch_id):
        from PIL import Image
        patch_dir = os.path.join(self.rawframes_dir, patch_id)
        imgs = []
        for i in range(1, self.clip_len+1):
            img_path = os.path.join(patch_dir, f'img_{i:04d}.jpg')
            img = Image.open(img_path).convert('RGB')
            imgs.append(img)
        return imgs

def update_teacher(student, teacher, ema_decay=0.99):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data = ema_decay * t.data + (1 - ema_decay) * s.data

def fix_x_shape(x, params):
    if x.ndim == 5 and x.shape[1] == 3:
        return x
    if x.ndim == 4 and x.shape[0] == params['bs'] * 3:
        B = params['bs']; C = 3; T = params['clip_len']; H = W = 224
        return x.view(B, C, T, H, W)
    if x.ndim == 5 and x.shape[0] == params['bs'] * 3:
        B = params['bs']; C = 3; T = params['clip_len']; H = W = 224
        return x.view(B, C, T, H, W)
    if x.ndim == 5 and x.shape[1] == params['clip_len'] and x.shape[2] == 3:
        return x.permute(0, 2, 1, 3, 4)
    raise ValueError(f"x shape not recognized: {x.shape}")

@torch.no_grad()
def evaluate_on_val(model, val_loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, return_loss=False)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.dim() == 5:
            logits = logits.mean(dim=[2, 3, 4])
        elif logits.dim() == 4:
            logits = logits.mean(dim=[2, 3])
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = total_loss / total if total > 0 else 0
    acc = correct / total if total > 0 else 0
    return acc, avg_loss

@torch.no_grad()
def evaluate_on_train(model, train_loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, return_loss=False)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.dim() == 5:
            logits = logits.mean(dim=[2, 3, 4])
        elif logits.dim() == 4:
            logits = logits.mean(dim=[2, 3])
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    acc = correct / total if total > 0 else 0
    return acc

def train_teacher_student(params):
    import copy
    from mmaction.apis import init_recognizer

    logger = setup_logger(params['save_dir'])
    save_config(params['config'], params['save_dir'], params)
    writer = SummaryWriter(log_dir=os.path.join(params['save_dir'], 'vis_data'))

    logger.info("Experiment started. Params: %s", str(params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = init_recognizer(params['config'], params['checkpoint'], device=device)
    teacher = init_recognizer(params['config'], params['checkpoint'], device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    consis_criterion = nn.MSELoss()
    optimizer = optim.SGD(
        student.parameters(),
        lr=params['lr'],
        momentum=0.9,
        weight_decay=1e-4)

    labeled_ds = LabeledDataset(
        params['train_csv'], params['rawframes_dir'],
        params['clip_len'], transform=get_transform())
    unlabeled_ds = UnlabeledDataset(
        params['unlabeled_csv'], params['rawframes_dir'],
        params['clip_len'], transform=get_transform())
    labeled_loader = DataLoader(
        labeled_ds, batch_size=params['bs'], shuffle=True,
        num_workers=4, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=params['bs'] * params['unlab_mult'],
        shuffle=True, num_workers=4, drop_last=True)

    val_csv = params.get('val_csv') or params['train_csv'].replace('train', 'val')
    val_ds = LabeledDataset(val_csv, params['rawframes_dir'], params['clip_len'], transform=get_transform())
    val_loader = DataLoader(val_ds, batch_size=params['bs'], shuffle=False, num_workers=4, drop_last=False)

    log_path = os.path.join(params['save_dir'], 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write('Epoch\tTrainLoss\tSupLoss\tConsisLoss\tTrainAcc\tValAcc\tValLoss\tLR\n')

    best_val_acc = 0.0

    for epoch in range(1, params['epochs'] + 1):
        student.train()
        unlabeled_iter = iter(unlabeled_loader)
        running_loss, running_sup, running_consis = 0, 0, 0
        batches = 0

        for x_l, y_l in labeled_loader:
            x_l = fix_x_shape(x_l, params).to(device)
            y_l = y_l.to(device)
            try:
                x_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u = next(unlabeled_iter)
            x_u = fix_x_shape(x_u, params).to(device)

            # ---- Forward student, logits only ----
            logits = student(x_l, return_loss=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if hasattr(logits, "logits"):
                logits = logits.logits
            if logits.dim() > 2:
                logits_flat = logits.mean(dim=list(range(2, logits.dim())))
            else:
                logits_flat = logits

            sup_loss = criterion(logits_flat, y_l)

            # ---- Consistency loss ----
            with torch.no_grad():
                t_logits_u = teacher(x_u, return_loss=False)
                if isinstance(t_logits_u, (tuple, list)):
                    t_logits_u = t_logits_u[0]
                if hasattr(t_logits_u, "logits"):
                    t_logits_u = t_logits_u.logits
                if t_logits_u.dim() > 2:
                    t_logits_u = t_logits_u.mean(dim=list(range(2, t_logits_u.dim())))
            s_logits_u = student(x_u, return_loss=False)
            if isinstance(s_logits_u, (tuple, list)):
                s_logits_u = s_logits_u[0]
            if hasattr(s_logits_u, "logits"):
                s_logits_u = s_logits_u.logits
            if s_logits_u.dim() > 2:
                s_logits_u = s_logits_u.mean(dim=list(range(2, s_logits_u.dim())))
            consis_loss = consis_criterion(s_logits_u, t_logits_u)

            loss = sup_loss + params['lambda_u'] * consis_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_teacher(student, teacher, ema_decay=params['ema_decay'])

            running_loss += loss.item()
            running_sup  += sup_loss.item()
            running_consis += consis_loss.item()
            batches += 1

        avg_loss = running_loss / batches
        avg_sup  = running_sup / batches
        avg_consis = running_consis / batches

        train_acc = evaluate_on_train(teacher, labeled_loader, device)
        val_acc, val_loss = evaluate_on_val(teacher, val_loader, device)

        # 日志、可视化、文件同步
        logger.info(f"[Epoch {epoch}] TrainLoss: {avg_loss:.4f} | SupLoss: {avg_sup:.4f} | "
                    f"ConsisLoss: {avg_consis:.4f} | TrainAcc: {train_acc:.4f} | "
                    f"ValAcc: {val_acc:.4f} | ValLoss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f"{epoch}\t{avg_loss:.4f}\t{avg_sup:.4f}\t"
                       f"{avg_consis:.4f}\t{train_acc:.4f}\t"
                       f"{val_acc:.4f}\t{val_loss:.4f}\t"
                       f"{optimizer.param_groups[0]['lr']:.2e}\n")
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/SupLoss', avg_sup, epoch)
        writer.add_scalar('Train/ConsisLoss', avg_consis, epoch)
        writer.add_scalar('Eval/TrainAcc', train_acc, epoch)
        writer.add_scalar('Eval/ValAcc', val_acc, epoch)
        writer.add_scalar('Eval/ValLoss', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if epoch % 10 == 0:
            torch.save(student.state_dict(), os.path.join(params['save_dir'], f'student_epoch_{epoch}.pth'))
            torch.save(teacher.state_dict(), os.path.join(params['save_dir'], f'teacher_epoch_{epoch}.pth'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(teacher.state_dict(), os.path.join(params['save_dir'], f'best_teacher_epoch_{epoch}.pth'))
            logger.info(f"*** New best model saved at epoch {epoch}, ValAcc={val_acc:.4f} ***")

    writer.close()
    torch.save(teacher.state_dict(), os.path.join(params['save_dir'], 'teacher_final.pth'))
    logger.info(f"Best teacher model saved to {os.path.join(params['save_dir'], 'teacher_final.pth')}")
    logger.info("Experiment finished.")


if __name__ == '__main__':
    assert len(sys.argv) == 2, "用法: python fix_checkpoint_swin.py configs/downstream_videoswin/xxx.py"
    cfg_path = sys.argv[1]
    cfg = load_cfg_py(cfg_path)
    params = extract_semi_params(cfg)

    # 自动生成带时间戳的保存目录
    from datetime import datetime

    cfg_base = os.path.splitext(os.path.basename(cfg_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = params['save_dir'] if 'save_dir' in params else 'work_dirs/teacher_student'
    new_dir = os.path.join(base_dir, f"{cfg_base}_{timestamp}")
    params['save_dir'] = new_dir

    train_teacher_student(params)