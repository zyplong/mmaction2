#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage: python train_swin_semi_ndvi_dualbranch.py \
           <rgb_swin_config.py> <paths_and_hyperparams_config.py>
"""
import os
import sys
import logging
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import pandas as pd
import tifffile
from mmaction.apis import init_recognizer

# ─── 动态加载任意 .py 配置 ─────────────────────────────────
def load_cfg_py(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

# ─── 提取所有参数 ──────────────────────────────────────────────
def extract_semi_params(cfg):
    p = {}
    p['train_csv']     = cfg.ann_file_train
    p['val_csv']       = getattr(cfg, 'ann_file_val', cfg.ann_file_train.replace('train','val'))
    p['unlabeled_csv'] = cfg.ann_file_unlabeled
    p['rgb_dir']       = cfg.data_prefix_rgb['img']
    p['ndvi_dir']      = cfg.data_prefix_ndvi['img']
    # clip_len & batch size
    clip = 8
    for op in cfg.train_dataloader['dataset']['pipeline']:
        if op.get('type')=='SampleFrames':
            clip = op.get('clip_len', clip)
    p['clip_len'] = clip
    p['bs']       = cfg.train_dataloader['batch_size']
    # lr, epochs, 半监督超参
    p['lr']        = cfg.optim_wrapper['optimizer']['lr']
    p['epochs']    = cfg.train_cfg['max_epochs']
    p['lambda_u']  = getattr(cfg, 'lambda_u', 1.0)
    p['ema_decay'] = getattr(cfg, 'ema_decay', 0.99)
    p['num_classes'] = cfg.model['cls_head']['num_classes']
    p['save_dir']    = cfg.work_dir
    return p

# ─── 日志 & 配置备份 ─────────────────────────────────────────────
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(save_dir,'train.log'), encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    return logger

def save_config(cfg_path, save_dir, params):
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(save_dir,'config_paths.py'))
    with open(os.path.join(save_dir,'run_params.txt'),'w',encoding='utf-8') as f:
        for k,v in params.items():
            f.write(f"{k}: {v}\n")

# ─── 有标签 Dataset ────────────────────────────────────────────────
class DualBranchLabeledDataset(Dataset):
    def __init__(self, ann_file, rgb_root, ndvi_root, clip_len, tf=None):
        df = pd.read_csv(ann_file, sep=r'\s+', header=None,
                         names=['pid','fid','label'])
        df = df[df['label'].astype(str).str.isdigit()]
        self.entries = df[['pid','label']].drop_duplicates().reset_index(drop=True)
        self.rgb_root, self.ndvi_root, self.clip_len = rgb_root, ndvi_root, clip_len
        self.tf = tf; self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid   = str(self.entries.loc[idx,'pid'])
        label = int(self.entries.loc[idx,'label'])
        rgb_f, ndv_f = [], []
        for t in range(1, self.clip_len+1):
            p1 = os.path.join(self.rgb_root, pid, f'img_{t:04d}.jpg')
            p2 = os.path.join(self.ndvi_root, pid, f'img_{t:04d}.tif')
            if not os.path.exists(p1): raise FileNotFoundError(f"RGB missing {p1}")
            if not os.path.exists(p2): raise FileNotFoundError(f"NDVI missing {p2}")

            # RGB
            im1 = Image.open(p1).convert('RGB')

            # NDVI 用 tifffile
            arr = tifffile.imread(p2)
            im2 = Image.fromarray(arr)
            if im2.mode != 'L':
                im2 = im2.convert('L')

            if self.tf:
                im1 = self.tf(im1)
                im2 = self.tf(im2)

            rgb_f.append(self.to_tensor(im1))
            ndv_f.append(self.to_tensor(im2))

        rgb  = torch.stack(rgb_f).permute(1,0,2,3)
        ndvi = torch.stack(ndv_f).permute(1,0,2,3)
        return rgb, ndvi, label

# ─── 无标签 Dataset ────────────────────────────────────────────────
class DualBranchUnlabeledDataset(Dataset):
    def __init__(self, ann_file, rgb_root, ndvi_root, clip_len, tf=None):
        df = pd.read_csv(ann_file, sep=r'\s+', header=None,
                         names=['pid','fid','label'])
        self.entries = df[['pid']].drop_duplicates().reset_index(drop=True)
        self.rgb_root, self.ndvi_root, self.clip_len = rgb_root, ndvi_root, clip_len
        self.tf = tf; self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid = str(self.entries.loc[idx,'pid'])
        rgb_f, ndv_f = [], []
        for t in range(1, self.clip_len+1):
            p1 = os.path.join(self.rgb_root, pid, f'img_{t:04d}.jpg')
            p2 = os.path.join(self.ndvi_root, pid, f'img_{t:04d}.tif')
            if not os.path.exists(p1): raise FileNotFoundError(f"RGB missing {p1}")
            if not os.path.exists(p2): raise FileNotFoundError(f"NDVI missing {p2}")

            im1 = Image.open(p1).convert('RGB')
            arr = tifffile.imread(p2)
            im2 = Image.fromarray(arr)
            if im2.mode != 'L':
                im2 = im2.convert('L')

            if self.tf:
                im1 = self.tf(im1)
                im2 = self.tf(im2)

            rgb_f.append(self.to_tensor(im1))
            ndv_f.append(self.to_tensor(im2))

        rgb  = torch.stack(rgb_f).permute(1,0,2,3)
        ndvi = torch.stack(ndv_f).permute(1,0,2,3)
        return rgb, ndvi

# ─── 双支路模型 ───────────────────────────────────────────────────
class DualBranchModel(nn.Module):
    def __init__(self, cfg_path, ckpt, num_classes, ndvi_ch=1, ndvi_dim=128):
        super().__init__()
        self.rgb_recog = init_recognizer(cfg_path, ckpt, device='cpu')
        feat_dim = self.rgb_recog.cls_head.fc_cls.in_features
        self.rgb_recog.cls_head = nn.Identity()
        self.ndvi_branch = nn.Sequential(
            nn.Conv3d(ndvi_ch,16,(2,3,3),padding=(0,1,1)),
            nn.ReLU(True), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(16,ndvi_dim,1),
            nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten()
        )
        self.fuse_fc = nn.Linear(feat_dim + ndvi_dim, num_classes)

    def forward(self, rgb, ndvi):
        device = next(self.parameters()).device
        rgb, ndvi = rgb.to(device), ndvi.to(device)
        f = self.rgb_recog.backbone(rgb).mean([2,3,4])
        g = self.ndvi_branch(ndvi)
        return self.fuse_fc(torch.cat([f,g],1))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for rgb, ndvi, y in loader:
        y = y.to(device)
        pred = model(rgb, ndvi).argmax(1)
        correct += (pred==y).sum().item()
        total   += y.size(0)
    return correct/total if total else 0

def update_teacher(student, teacher, ema):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(ema).add_(sp.data, alpha=1-ema)

def train_teacher_student(params, rgb_cfg_path):
    logger = setup_logger(params['save_dir'])
    save_config(rgb_cfg_path, params['save_dir'], params)
    writer = SummaryWriter(os.path.join(params['save_dir'],'tb'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # student / teacher
    student = DualBranchModel(rgb_cfg_path, params['checkpoint'], params['num_classes']).to(device)
    teacher = DualBranchModel(rgb_cfg_path, params['checkpoint'], params['num_classes']).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    sup_cr = nn.CrossEntropyLoss()
    con_cr = nn.MSELoss()
    optim_ = optim.SGD(student.parameters(), lr=params['lr'], momentum=0.9, weight_decay=1e-4)
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip()])

    L = DualBranchLabeledDataset(params['train_csv'],   params['rgb_dir'], params['ndvi_dir'], params['clip_len'], tf)
    U = DualBranchUnlabeledDataset(params['unlabeled_csv'], params['rgb_dir'], params['ndvi_dir'], params['clip_len'], tf)
    V = DualBranchLabeledDataset(params['val_csv'],     params['rgb_dir'], params['ndvi_dir'], params['clip_len'], tf)

    Ld = DataLoader(L, batch_size=params['bs'], shuffle=True,  drop_last=True,  num_workers=0)
    Ud = DataLoader(U, batch_size=params['bs'], shuffle=True,  drop_last=True,  num_workers=0)
    Vd = DataLoader(V, batch_size=params['bs'], shuffle=False, drop_last=False, num_workers=0)

    print(f"[Debug] L={len(L)}, U={len(U)}, V={len(V)}")
    best_val = 0.0

    for ep in range(1, params['epochs']+1):
        student.train()
        itU = iter(Ud); sum_sup = sum_con = 0.0
        for rgb_l, ndv_l, y_l in Ld:
            rgb_l, ndv_l, y_l = rgb_l.to(device), ndv_l.to(device), y_l.to(device)
            try:
                rgb_u, ndv_u = next(itU)
            except StopIteration:
                itU = iter(Ud); rgb_u, ndv_u = next(itU)
            rgb_u, ndv_u = rgb_u.to(device), ndv_u.to(device)

            out_l = student(rgb_l, ndv_l); ls = sup_cr(out_l, y_l)
            with torch.no_grad():
                to = teacher(rgb_u, ndv_u)
            so = student(rgb_u, ndv_u); lc = con_cr(so, to)

            loss = ls + params['lambda_u'] * lc
            optim_.zero_grad(); loss.backward(); optim_.step()
            update_teacher(student, teacher, params['ema_decay'])

            sum_sup += ls.item(); sum_con += lc.item()

        tr_acc = evaluate(teacher, Ld, device)
        va_acc = evaluate(teacher, Vd, device)
        logger.info(f"[{ep}] Sup={sum_sup/len(Ld):.4f} Con={sum_con/len(Ld):.4f} Tr={tr_acc:.4f} Val={va_acc:.4f}")
        writer.add_scalar('Val/Acc', va_acc, ep)
        if va_acc > best_val:
            best_val = va_acc
            torch.save(teacher.state_dict(), os.path.join(params['save_dir'],'best.pth'))

    writer.close()
    logger.info(f"Done! Best ValAcc={best_val:.4f}")

if __name__ == '__main__':
    assert len(sys.argv)==3, "Usage: python train_swin_semi_ndvi_dualbranch.py <rgb_swin_config.py> <paths_and_hyperparams_config.py>"
    rgb_cfg_path   = sys.argv[1]
    paths_cfg_path = sys.argv[2]

    paths_cfg = load_cfg_py(paths_cfg_path)
    params    = extract_semi_params(paths_cfg)

    rgb_cfg = load_cfg_py(rgb_cfg_path)
    if 'rgb_backbone' in rgb_cfg.model:
        params['checkpoint'] = rgb_cfg.model['rgb_backbone']['pretrained']
    elif 'backbone' in rgb_cfg.model:
        params['checkpoint'] = rgb_cfg.model['backbone']['pretrained']
    else:
        raise KeyError("找不到 pretrained checkpoint，请检查 rgb config")

    t0 = datetime.now().strftime("%Y%m%d_%H%M%S")
    params['save_dir'] = os.path.join(params['save_dir'], f"dual_{t0}")

    train_teacher_student(params, rgb_cfg_path)