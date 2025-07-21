# inference_save_pred.py
import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from mmaction.apis import init_recognizer

# —— Monkey-patch PatchEmbed3D 使其能自动补齐缺失的 batch/channel 维度 ——
from mmaction.models.backbones.swin import PatchEmbed3D
_orig_pe_forward = PatchEmbed3D.forward
def _patched_pe_forward(self, x):
    # 如果传进来的是 4D tensor（比如 [C,D,H,W] 或 [B,D,H,W]），就补齐到 5D
    if x.ndim == 4:
        # 如果第 0 维就是通道数，说明是 [C,D,H,W]，做 batch 维补齐
        if x.shape[0] == self.in_channels:
            x = x.unsqueeze(0)           # -> [1,C,D,H,W]
        else:
            # 否则认为是 [B,D,H,W]，做 channel 维补齐
            x = x.unsqueeze(1)           # -> [B,1,D,H,W]
    return _orig_pe_forward(self, x)
PatchEmbed3D.forward = _patched_pe_forward
# ————————————————————————————————————————————————

def parse_args():
    p = argparse.ArgumentParser(
        description="用视频教师模型对无标签 patches 做前向，保存每个 patch 的 logits.npy")
    p.add_argument('--config',     required=True,
                   help='MMA2 视频识别 cfg 文件路径')
    p.add_argument('--checkpoint', required=True,
                   help='教师模型权重 .pth')
    p.add_argument('--rawframes',  required=True,
                   help='原始帧根目录，比如 datasets/rawframes')
    p.add_argument('--list',       required=True,
                   help='无标签 patch 列表，每行一个 patch_id')
    p.add_argument('--out_dir',    default='results/logits',
                   help='保存 logits 的目录')
    return p.parse_args()

def load_frames(patch_dir, clip_len=8):
    """
    从 patch_dir/img_0001.jpg…img_{clip_len:04d}.jpg
    读到一个 Tensor [1,3,T,H,W]
    """
    img_paths = [os.path.join(patch_dir, f'img_{i:04d}.jpg')
                 for i in range(1, clip_len+1)]
    imgs = [Image.open(p).convert('RGB') for p in img_paths]
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],
            std =[ 58.395/255,  57.12/255,  57.375/255],
        ),
    ])
    vids = [tf(im) for im in imgs]               # list of [3,H,W]
    video = torch.stack(vids, dim=1).unsqueeze(0) # -> [1,3,T,H,W]
    return video

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_recognizer(args.config, args.checkpoint, device=device)
    model.eval()

    # 2. 读 patch_id 列表
    with open(args.list, 'r') as f:
        patch_ids = [l.strip() for l in f if l.strip()]

    # 3. 遍历 inference 并保存
    for pid in patch_ids:
        patch_dir = os.path.join(args.rawframes, pid)
        if not os.path.isdir(patch_dir):
            print(f"[WARN] 找不到目录 {patch_dir}，跳过")
            continue

        video = load_frames(patch_dir).to(device)
        #（可选）检查维度
        #print("DEBUG video.shape =", video.shape)

        with torch.no_grad():
            logits = model(video, return_loss=False)
            # 解包各种 tuple/.logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if hasattr(logits, 'logits'):
                logits = logits.logits
            # 如果还有额外的时空维度，做全局平均池化到 [1,C]
            if logits.dim() > 2:
                logits = logits.mean(dim=list(range(2, logits.dim())))
            logits = logits.squeeze(0)  # -> [C]

        out_path = os.path.join(args.out_dir, f'{pid}.npy')
        np.save(out_path, logits.cpu().numpy())
        print(f"[SAVE] {out_path}")

if __name__ == '__main__':
    main()