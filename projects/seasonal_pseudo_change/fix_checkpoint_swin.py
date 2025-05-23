# import torch
#
# # 你的原始权重路径
# src = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/pretrained/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth'
#
# # 保存新权重的路径
# dst = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/pretrained/videoswin/swin_tiny_fixed.pth'
#
# ckpt = torch.load(src, map_location='cpu')
#
# # 如果不是 dict(model=...) 结构，则添加 model 键
# if 'model' not in ckpt:
#     if 'state_dict' in ckpt:
#         new_ckpt = {'model': ckpt['state_dict']}
#     else:
#         new_ckpt = {'model': ckpt}
#     torch.save(new_ckpt, dst)
#     print(f'[OK] Fixed checkpoint saved to {dst}')
# else:
#     print('[SKIP] Checkpoint already has "model" key')



# import pandas as pd
#
# val_csv = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.csv'
# val_txt = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt'
# df = pd.read_csv(val_csv)
# df.to_csv(val_txt, sep=' ', index=False, header=False)

# import torch
#
# # 加载原始裸 state_dict
# path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\videoswin\swin_tiny_patch244_window877_kinetics400_1k.pth'
# state_dict = torch.load(path)
#
# # 包装成 Swin3D 需要的格式
# wrapped = {'model': state_dict}
#
# # 保存为新的 .pth
# save_path = path.replace('.pth', '_converted.pth')
# torch.save(wrapped, save_path)
#
# print(f'保存成功: {save_path}')

import torch
import os

def convert(src_path: str, dst_path: str = None):
    # load original checkpoint
    ckpt = torch.load(src_path, map_location='cpu')
    # 有些权重是直接 dict，有些是 {'state_dict': ...}，还有可能是 {'model': ...}
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt

    # 去掉最外层可能的 "backbone." 前缀
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('backbone.'):
            nk = k[len('backbone.'):]
        else:
            nk = k
        new_sd[nk] = v

    # MMAction2 期待 checkpoint 的最外层有个 'model' key
    wrapped = {'model': new_sd}

    # 如果没传 dst，就默认在 src 同目录下加上 _converted
    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = base + '_converted' + ext

    torch.save(wrapped, dst_path)
    print(f'✅ Converted checkpoint saved to: {dst_path}')

# ———————— 这里开始，直接写死你本地的文件路径 ————————
if __name__ == '__main__':
    # 请根据你的实际路径修改下面两个变量
    src_path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\videoswin\swin_tiny_patch244_window877_kinetics400_1k.pth'
    dst_path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\videoswin\swin_tiny_patch244_window877_kinetics400_1k_converted.pth'

    # 调用转换函数
    convert(src_path, dst_path)