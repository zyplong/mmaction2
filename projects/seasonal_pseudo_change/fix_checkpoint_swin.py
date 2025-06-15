# fix_checkpoint_swin.py

# 1) 最顶端，加补丁
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
    return params

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, rawframes_dir, clip_len=8, transform=None):
        self.df = pd.read_csv(csv_file, header=None, names=['patch_id', 'frame_num', 'label'])
        # 过滤掉 'label' 行（表头）
        self.df = self.df[self.df['label'] != 'label']
        self.rawframes_dir = rawframes_dir
        self.clip_len = clip_len
        self.transform = transform
        self.df = self.df.drop_duplicates(subset=['patch_id', 'label'])

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
    # 确保 x 是 (B,3,T,H,W)
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

def train_teacher_student(params):
    import copy
    from mmaction.apis import init_recognizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(params['save_dir'], exist_ok=True)

    student = init_recognizer(
        params['config'], params['checkpoint'], device=device)
    teacher = init_recognizer(
        params['config'], params['checkpoint'], device=device)
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

    for epoch in range(1, params['epochs'] + 1):
        student.train()
        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in labeled_loader:
            x_l = fix_x_shape(x_l, params).to(device)
            y_l = y_l.to(device)

            # 每次都拿一个无标签batch
            try:
                x_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u = next(unlabeled_iter)
            x_u = fix_x_shape(x_u, params).to(device)

            # 有监督loss
            sup_loss = student(x_l, y_l, return_loss=True)
            if isinstance(sup_loss, dict):
                sup_loss = sup_loss['loss_cls']
            if hasattr(sup_loss, "dim") and sup_loss.dim() > 0:
                sup_loss = sup_loss.mean()

            # 一致性loss
            with torch.no_grad():
                t_logits_u = teacher(x_u, return_loss=False)
            s_logits_u = student(x_u, return_loss=False)
            consis_loss = consis_criterion(s_logits_u, t_logits_u)
            if hasattr(consis_loss, "dim") and consis_loss.dim() > 0:
                consis_loss = consis_loss.mean()

            loss = sup_loss + params['lambda_u'] * consis_loss
            # print(f"DEBUG: sup_loss: {sup_loss}, consis_loss: {consis_loss}, loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_teacher(student, teacher, ema_decay=params['ema_decay'])

        print(
            f"Epoch {epoch} | "
            f"SupLoss {sup_loss.item():.4f} | "
            f"ConsisLoss {consis_loss.item():.4f} | "
            f"LR {optimizer.param_groups[0]['lr']:.6f}"
        )
        if epoch % 10 == 0:
            torch.save(
                student.state_dict(),
                os.path.join(params['save_dir'], f'student_epoch_{epoch}.pth'))
            torch.save(
                teacher.state_dict(),
                os.path.join(params['save_dir'], f'teacher_epoch_{epoch}.pth'))

    torch.save(
        teacher.state_dict(),
        os.path.join(params['save_dir'], 'teacher_final.pth'))
    print(f"Best teacher model saved to {os.path.join(params['save_dir'], 'teacher_final.pth')}")

if __name__ == '__main__':
    assert len(sys.argv) == 2, "用法: python fix_checkpoint_swin.py configs/downstream_videoswin/xxx.py"
    cfg_path = sys.argv[1]
    cfg = load_cfg_py(cfg_path)
    params = extract_semi_params(cfg)
    print('Train params:', params)
    train_teacher_student(params)