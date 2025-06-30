#
# from mmaction.models.backbones.swin import PatchEmbed3D
# _PatchEmbed3D_forward = PatchEmbed3D.forward
# def _patched_forward(self, x):
#     if x.ndim == 4:
#         BD, D, H, W = x.shape
#         C = self.in_channels
#         B = BD // C
#         x = x.view(B, C, D, H, W)
#     return _PatchEmbed3D_forward(self, x)
# PatchEmbed3D.forward = _patched_forward
#
# import os
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from mmaction.apis import init_recognizer
# from PIL import Image
#
# config_file = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\config.py'
# checkpoint_file = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\teacher_final.pth'
# rawframes_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
# val_csv_path = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\val.txt'
# save_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\pred.csv'
# clip_len = 20
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = init_recognizer(config_file, checkpoint_file, device=device)
# model.eval()
#
# df = pd.read_csv(val_csv_path, sep=' ', header=None, names=['patch_id', 'frame_num', 'label'])
# rows = []
#
# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     patch_id = row['patch_id']
#     y_true = int(row['label'])
#     imgs = []
#     for i in range(1, clip_len + 1):
#         img_path = os.path.join(rawframes_dir, patch_id, f'img_{i:04d}.jpg')
#         if not os.path.exists(img_path):
#             print(f"Warning: {img_path} not found!")
#             continue
#         img = Image.open(img_path).convert('RGB').resize((224, 224))
#         img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.
#         imgs.append(img)
#     if len(imgs) != clip_len:
#         print(f"Patch {patch_id} only has {len(imgs)} frames, skip.")
#         continue
#     video = torch.stack(imgs)  # [T, 3, 224, 224]
#     video = video.permute(1, 0, 2, 3)  # [3, T, 224, 224]
#     video = video.unsqueeze(0).to(device)  # [1, 3, T, 224, 224]
#     print(f"{patch_id}: video shape {video.shape}")
#
#     with torch.no_grad():
#         logits = model(video, return_loss=False)
#         if isinstance(logits, (tuple, list)):
#             logits = logits[0]
#         if hasattr(logits, "logits"):
#             logits = logits.logits
#         if logits.dim() > 2:
#             logits = logits.mean(dim=list(range(2, logits.dim())))
#         pred = logits.argmax(dim=1).item()
#     rows.append([patch_id, y_true, pred])
#
# pd.DataFrame(rows, columns=['patch_id', 'y_true', 'y_pred']).to_csv(save_csv, index=False)
# print(f'已保存预测结果到 {save_csv}')





