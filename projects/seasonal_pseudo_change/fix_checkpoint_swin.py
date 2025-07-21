#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mean Teacher 半监督 (Step0 升级版，修复 val.txt 加载)
Video Swin + 强/弱增广 + Teacher EMA 双阶段 + 置信过滤 + 温度锐化 + ramp-up
"""
import os, sys, math, shutil, logging, importlib.util, random
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

from mmaction.apis import init_recognizer
from mmaction.models.backbones.swin import PatchEmbed3D

# —————— Patch Swin PatchEmbed3D 支持 4D 输入 ——————
_orig_forward = PatchEmbed3D.forward
def _patched_forward(self, x):
    if x.ndim == 4:
        BD, D, H, W = x.shape
        C = self.in_channels
        B = BD // C
        x = x.view(B, C, D, H, W)
    return _orig_forward(self, x)
PatchEmbed3D.forward = _patched_forward

# —————— 配置加载 & 日志 ——————
def load_cfg_py(cfg_path):
    spec = importlib.util.spec_from_file_location("cfg", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["cfg"] = cfg
    spec.loader.exec_module(cfg)
    return cfg

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

def save_run_info(cfg_path, save_dir, params):
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(save_dir, "config.py"))
    with open(os.path.join(save_dir, "params.txt"), "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def extract_params(cfg):
    p = {}
    p['config'] = cfg.__file__
    p['checkpoint'] = cfg.model['backbone'].get('pretrained', None)
    # train_val_csv 支持 .csv 或 .txt（自动转成 .csv 或用 sep=None）
    train = cfg.ann_file_train
    p['train_csv'] = train.replace('.txt', '.csv') if train.endswith('.txt') else train
    val = getattr(cfg, 'ann_file_val', None) or p['train_csv'].replace('train', 'val')
    p['val_csv'] = val.replace('.txt', '.csv') if val.endswith('.txt') else val
    p['unlabeled_txt'] = getattr(cfg, 'ann_file_unlabeled', None)
    p['rawframes_dir'] = cfg.data_root
    sample = cfg.train_dataloader['dataset']['pipeline'][0]
    p['clip_len'] = sample.get('clip_len', 8)
    p['bs'] = cfg.train_dataloader.get('batch_size', 8)
    p['lr'] = cfg.optim_wrapper['optimizer'].get('lr', 1e-3)
    p['epochs'] = cfg.train_cfg.get('max_epochs', 50)
    p['save_dir'] = cfg.work_dir if hasattr(cfg, 'work_dir') else 'work_dirs/mean_teacher'
    # 半监督专用
    p.update({
        'lambda_u': 2.0,
        'rampup_epochs': 5,
        'tau_start': 0.95,
        'tau_end': 0.85,
        'tau_warm': 3,
        'T_sharp': 0.5,
        'ema_decay_base': 0.99,
        'ema_decay_late': 0.995,
        'ema_switch_epoch': 10,
        'num_classes': cfg.model['cls_head']['num_classes']
    })
    return p

# —————— 数据统一与增广 ——————
def fix_x_shape(x, params):
    # 期望 [B,3,T,H,W]
    if x.ndim == 5 and x.shape[1] == 3:
        return x
    if x.ndim == 5 and x.shape[2] == 3:
        return x.permute(0,2,1,3,4)
    raise ValueError(f"Unexpected input shape {x.shape}")

def weak_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

def strong_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, raw_dir, clip_len, w_aug):
        # 支持 .csv & .txt
        self.df = pd.read_csv(ann_file, sep=None, engine='python',
                              header=None, names=['pid','frame','label'])
        self.df = self.df[self.df['label'].astype(str).str.isdigit()]
        self.df = self.df.drop_duplicates(['pid','label'])
        self.raw_dir, self.clip_len, self.w_aug = raw_dir, clip_len, w_aug
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        pid = str(self.df.iloc[idx]['pid'])
        label = int(self.df.iloc[idx]['label'])
        imgs = []
        for i in range(1, self.clip_len+1):
            p = os.path.join(self.raw_dir, pid, f'img_{i:04d}.jpg')
            imgs.append(Image.open(p).convert('RGB'))
        w = [self.w_aug(im) for im in imgs]
        video = torch.stack(w).permute(1,0,2,3)  # [3,T,H,W]
        return video, label

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, raw_dir, clip_len, w_aug, s_aug):
        with open(txt_file) as f:
            self.pids = [l.strip() for l in f if l.strip()]
        self.raw_dir, self.clip_len = raw_dir, clip_len
        self.w_aug, self.s_aug = w_aug, s_aug
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, idx):
        pid = self.pids[idx]
        imgs = []
        for i in range(1, self.clip_len+1):
            p = os.path.join(self.raw_dir, pid, f'img_{i:04d}.jpg')
            imgs.append(Image.open(p).convert('RGB'))
        w = [self.w_aug(im) for im in imgs]
        s = [self.s_aug(im) for im in imgs]
        vw = torch.stack(w).permute(1,0,2,3)
        vs = torch.stack(s).permute(1,0,2,3)
        return vw, vs

# —————— 调度函数 ——————
def linear_rampup(epoch, rampup_length):
    if rampup_length == 0: return 1.0
    return float(np.clip(epoch / rampup_length, 0., 1.))

def tau_schedule(epoch, p):
    if epoch <= p['tau_warm']:
        return p['tau_start']
    total = max(1, p['epochs'] - p['tau_warm'])
    prog = min(1.0, (epoch - p['tau_warm']) / total)
    return p['tau_start'] + (p['tau_end'] - p['tau_start']) * prog

def ema_decay_schedule(epoch, p):
    return p['ema_decay_base'] if epoch < p['ema_switch_epoch'] else p['ema_decay_late']

@torch.no_grad()
def update_teacher(student, teacher, decay):
    for t,s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1-decay)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0,0,0.0
    ce = nn.CrossEntropyLoss()
    for x,y in loader:
        x = fix_x_shape(x, {'bs':x.size(0),'clip_len':x.shape[2]})
        x,y = x.to(device), y.to(device)
        out = model(x, return_loss=False)
        if isinstance(out,(tuple,list)):
            out = out[0]
        if hasattr(out,'logits'):
            out = out.logits
        if out.dim()>2:
            out = out.mean(dim=list(range(2,out.dim())))
        loss = ce(out,y)
        loss_sum += loss.item()*x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    acc = correct/total if total>0 else 0
    return acc, (loss_sum/total if total>0 else 0)

def train_mean_teacher(params):
    # 准备
    base = os.path.splitext(os.path.basename(params['config']))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(params['save_dir'], f"{base}_{ts}")
    params['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    save_run_info(params['config'], save_dir, params)
    writer = SummaryWriter(os.path.join(save_dir,'tb'))
    logger.info("Params:\n%s", params)

    # 随机种子
    seed=42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = init_recognizer(params['config'], params['checkpoint'], device=device)
    teacher = init_recognizer(params['config'], params['checkpoint'], device=device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad=False

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='none')
    optimizer = optim.SGD(student.parameters(),
                          lr=params['lr'], momentum=0.9, weight_decay=1e-4)

    # Loader
    w_aug = weak_transform(); s_aug = strong_transform()
    L_ds = LabeledDataset(params['train_csv'], params['rawframes_dir'],
                          params['clip_len'], w_aug)
    V_ds = LabeledDataset(params['val_csv'],   params['rawframes_dir'],
                          params['clip_len'], w_aug)
    U_ds = UnlabeledDataset(params['unlabeled_txt'], params['rawframes_dir'],
                            params['clip_len'], w_aug, s_aug)

    L_loader = DataLoader(L_ds, batch_size=params['bs'], shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
    U_loader = DataLoader(U_ds, batch_size=params['bs'], shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
    V_loader = DataLoader(V_ds, batch_size=params['bs'], shuffle=False,
                          num_workers=4, drop_last=False)

    logger.info(f"Dataset sizes L={len(L_ds)}, U={len(U_ds)}, V={len(V_ds)}")

    best_val=0.0
    logf = open(os.path.join(save_dir,'log.tsv'),'w',encoding='utf-8')
    logf.write("epoch\tsup\tunsup\tmask%\ttrain_acc\tval_acc\tval_loss\n")

    unl_iter = iter(U_loader)
    for epoch in range(1, params['epochs']+1):
        student.train()
        sup_sum = unsup_sum = mask_cnt = total_cnt = 0
        pbar = tqdm(L_loader, desc=f"Epoch {epoch}/{params['epochs']}", ncols=110)

        tau_e = tau_schedule(epoch, params)
        ema_d = ema_decay_schedule(epoch, params)
        lam_u = params['lambda_u'] * linear_rampup(epoch, params['rampup_epochs'])

        for x_l, y_l in pbar:
            x_l = fix_x_shape(x_l, params).to(device)
            y_l = y_l.to(device)
            try:
                x_uw, x_us = next(unl_iter)
            except StopIteration:
                unl_iter = iter(U_loader)
                x_uw, x_us = next(unl_iter)
            x_uw = fix_x_shape(x_uw, params).to(device)
            x_us = fix_x_shape(x_us, params).to(device)

            # supervised
            s_out_l = student(x_l, return_loss=False)
            if isinstance(s_out_l,(tuple,list)): s_out_l=s_out_l[0]
            if hasattr(s_out_l,'logits'): s_out_l=s_out_l.logits
            if s_out_l.dim()>2:
                s_out_l = s_out_l.mean(dim=list(range(2,s_out_l.dim())))
            sup_loss = ce_loss(s_out_l, y_l)

            # teacher pseudo
            with torch.no_grad():
                t_out = teacher(x_uw, return_loss=False)
                if isinstance(t_out,(tuple,list)): t_out=t_out[0]
                if hasattr(t_out,'logits'): t_out=t_out.logits
                if t_out.dim()>2:
                    t_out = t_out.mean(dim=list(range(2,t_out.dim())))
                t_prob = F.softmax(t_out/params['T_sharp'], dim=1)
                maxp, pseudo = t_prob.max(dim=1)
                mask = (maxp >= tau_e).float()

            # student consistency
            s_out_u = student(x_us, return_loss=False)
            if isinstance(s_out_u,(tuple,list)): s_out_u=s_out_u[0]
            if hasattr(s_out_u,'logits'): s_out_u=s_out_u.logits
            if s_out_u.dim()>2:
                s_out_u = s_out_u.mean(dim=list(range(2,s_out_u.dim())))
            logp_s = F.log_softmax(s_out_u, dim=1)
            kl_all = kl_loss(logp_s, t_prob).sum(dim=1)
            if mask.sum()>0:
                unsup_loss = (kl_all*mask).sum()/mask.sum()
            else:
                unsup_loss = torch.zeros((), device=device)

            loss = sup_loss + lam_u * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_teacher(student, teacher, decay=ema_d)

            sup_sum += sup_loss.item()
            unsup_sum += unsup_loss.item()
            mask_cnt += mask.sum().item()
            total_cnt += mask.numel()

            pbar.set_postfix({
                'sup':f"{sup_loss.item():.3f}",
                'unsup':f"{unsup_loss.item():.3f}",
                'mask%':f"{100*mask.mean().item():.1f}",
                'λ_u':f"{lam_u:.2f}",
                'τ':f"{tau_e:.2f}"
            })

        # eval
        train_acc,_ = evaluate(teacher, L_loader, device)
        val_acc, val_loss = evaluate(teacher, V_loader, device)
        avg_sup = sup_sum/len(L_loader)
        avg_unsup = unsup_sum/len(L_loader)
        mask_ratio = mask_cnt/max(1,total_cnt)

        logger.info(f"[Epoch {epoch}] sup={avg_sup:.4f} unsup={avg_unsup:.4f} "
                    f"mask={mask_ratio:.3f} train_acc={train_acc:.4f} "
                    f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
                    f"τ={tau_e:.2f} λ_u={lam_u:.2f} ema={ema_d:.3f}")

        logf.write(f"{epoch}\t{avg_sup:.4f}\t{avg_unsup:.4f}\t"
                   f"{mask_ratio:.3f}\t{train_acc:.4f}\t"
                   f"{val_acc:.4f}\t{val_loss:.4f}\n")
        logf.flush()

        # tensorboard
        writer.add_scalar('Loss/Sup', avg_sup, epoch)
        writer.add_scalar('Loss/Unsup', avg_unsup, epoch)
        writer.add_scalar('Mask/Ratio', mask_ratio, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Sched/tau', tau_e, epoch)
        writer.add_scalar('Sched/lambda_u', lam_u, epoch)
        writer.add_scalar('Sched/ema_decay', ema_d, epoch)

        # 保存
        if val_acc > best_val:
            best_val = val_acc
            torch.save(teacher.state_dict(),
                       os.path.join(save_dir, f"best_teacher_ep{epoch}.pth"))
            logger.info(f"*** New best teacher (ep{epoch}, val_acc={val_acc:.4f}) ***")
        if epoch % 10 == 0:
            torch.save(student.state_dict(),
                       os.path.join(save_dir, f"student_ep{epoch}.pth"))
            torch.save(teacher.state_dict(),
                       os.path.join(save_dir, f"teacher_ep{epoch}.pth"))

    torch.save(teacher.state_dict(), os.path.join(save_dir, "teacher_final.pth"))
    torch.save(student.state_dict(), os.path.join(save_dir, "student_final.pth"))
    logf.close(); writer.close()
    logger.info("Training finished. Best val_acc = %.4f", best_val)

if __name__ == "__main__":
    assert len(sys.argv)==2, "用法: python train_mean_teacher_step0_fixed.py <config.py>"
    cfg = load_cfg_py(sys.argv[1])
    params = extract_params(cfg)
    train_mean_teacher(params)