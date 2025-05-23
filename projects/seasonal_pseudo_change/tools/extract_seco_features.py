# tools/extract_seco_features.py

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from moco2_module import MoCoV2ResNet50

def main():
    # ——— 1. 配置：把下面的三个路径改成你自己机器上真实存在的路径 ———
    ckpt_path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\seco\seco_resnet50_1m.ckpt'
    raw_dir   = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
    out_dir   = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\npy_features'
    # ——————————————————————————————————————————————————————————————

    # 1. 路径检查
    if not os.path.isfile(ckpt_path):
        print(f'Error: checkpoint not found:\n  {ckpt_path}')
        sys.exit(1)
    if not os.path.isdir(raw_dir):
        print(f'Error: rawframes directory not found:\n  {raw_dir}')
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # 2. 加载模型
    print('Loading SeCo model and checkpoint...')
    model = MoCoV2ResNet50()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval().cuda()

    # 3. 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 4. 提取特征
    print('Start feature extraction...')
    for patch in sorted(os.listdir(raw_dir)):
        pdir = os.path.join(raw_dir, patch)
        if not os.path.isdir(pdir):
            continue

        feats = []
        for imgname in sorted(os.listdir(pdir)):
            if not imgname.lower().endswith('.jpg'):
                continue
            img_path = os.path.join(pdir, imgname)
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                feat = model(x).cpu().squeeze()
            feats.append(feat.numpy())

        arr = np.stack(feats, axis=0)  # [N_frames, 2048]
        save_path = os.path.join(out_dir, f'{patch}.npy')
        np.save(save_path, arr)
        print(f'Saved {save_path} → shape {arr.shape}')

    print('Done! All patches processed.')

if __name__ == '__main__':
    main()


    