import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
from torchvision import transforms
from downstream.models.seco_resnet import SeCoResNet







def main():
    # ——— 1. 配置路径 ———
    ckpt_path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\seco\seco_resnet50_1m.ckpt'
    raw_dir   = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
    out_dir   = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\pt_spatial_features'

    # ——— 2. 路径检查 ———
    if not os.path.isfile(ckpt_path):
        print(f'Error: checkpoint not found:\n  {ckpt_path}')
        sys.exit(1)
    if not os.path.isdir(raw_dir):
        print(f'Error: rawframes directory not found:\n  {raw_dir}')
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # ——— 3. 加载模型 ———
    print('Loading SeCo model and checkpoint...')
    model = SeCoResNet()  # ✅ 使用你定义的 SecoResNet 类（输出 (B,2048,4,4)）
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval().cuda()

    # ——— 4. 图像预处理 ———
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ——— 5. 特征提取 ———
    print('Start feature extraction (spatial)...')
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
                feat_map = model(x)  # (1, 2048, 4, 4)
                feat_map = feat_map.squeeze(0).cpu()  # (2048, 4, 4)
            feats.append(feat_map)

        tensor = torch.stack(feats, dim=0)  # (T, 2048, 4, 4)
        save_path = os.path.join(out_dir, f'{patch}.pt')
        torch.save(tensor, save_path)
        print(f'Saved {save_path} → shape {tuple(tensor.shape)}')

    print('Done! All patches processed.')

if __name__ == '__main__':
    main()
