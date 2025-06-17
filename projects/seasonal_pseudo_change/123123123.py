# # import os
# # import shutil
# #
# # years = ['2019', '2020', '2021', '2022', '2023']
# # seasons = ['spring', 'summer', 'autumn', 'winter']
# # src_root = r"F:\zyp\数据文件夹\1000"
# # dst_root = r"F:\zyp\数据文件夹\1000_renamed"   # 新建目录避免覆盖
# #
# # os.makedirs(dst_root, exist_ok=True)
# #
# # for year in years:
# #     src_dir = os.path.join(src_root, f"gee_farmland_{year}")
# #     dst_dir = os.path.join(dst_root, f"gee_farmland_{year}")
# #     os.makedirs(dst_dir, exist_ok=True)
# #     for patch_id in range(1, 1001):
# #         new_patch_id = patch_id + 300
# #         for season in seasons:
# #             src_name = f"patch_{patch_id}_{season}.tif"
# #             dst_name = f"patch_{new_patch_id}_{season}.tif"
# #             src_path = os.path.join(src_dir, src_name)
# #             dst_path = os.path.join(dst_dir, dst_name)
# #             if os.path.exists(src_path):
# #                 shutil.copyfile(src_path, dst_path)
# #                 print(f"{src_path} -> {dst_path}")
# #             else:
# #                 print(f"❌ 缺失: {src_path}")
# # print("🎉 批量重命名（复制）完成！")
#
#
#
#
#
#
#
# import os, re, numpy as np, tifffile
# from PIL import Image
#
# # === 路径（已适配你的实际目录）===
# PROJ_ROOT = r"F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change"
# SRC_ROOT  = r"F:\zyp\数据文件夹\1000_renamed"      # 你的重命名tif文件夹
# DST_ROOT  = os.path.join(PROJ_ROOT, "datasets", "rawframes")  # 输出 jpg 目录
# os.makedirs(DST_ROOT, exist_ok=True)
#
# # === 文件名规则 ===
# season_order = ['spring', 'summer', 'autumn', 'winter']
# pat = re.compile(r"patch_(\d+)_(spring|summer|autumn|winter)\.tif$", re.I)
#
# # === 百分位拉伸函数 ===
# def stretch_uint8(arr16, p_low=1, p_high=99):
#     arr = arr16.astype(np.float32)
#     lo, hi = np.percentile(arr, [p_low, p_high])
#     arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255
#     return arr.astype(np.uint8)
#
# # === 遍历所有年份 ===
# year_dirs = sorted(
#     [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))],
#     key=lambda d: d.split('_')[-1]
# )
# year2idx = {d: i for i, d in enumerate(year_dirs)}
#
# # === 主转换循环 ===
# for patch_id in range(301, 1301):  # 只处理 patch_301~patch_1300
#     patch_dir = os.path.join(DST_ROOT, f"patch_{patch_id}")
#     os.makedirs(patch_dir, exist_ok=True)
#     for ydir in year_dirs:
#         yidx = year2idx[ydir]
#         ypath = os.path.join(SRC_ROOT, ydir)
#         for sidx, season in enumerate(season_order):
#             tif_name = f"patch_{patch_id}_{season}.tif"
#             tif_path = os.path.join(ypath, tif_name)
#             jpg_name = f"img_{yidx * 4 + sidx + 1:04d}.jpg"
#             if not os.path.exists(tif_path):
#                 print(f"[缺失] {tif_path}")
#                 continue
#             try:
#                 arr = tifffile.imread(tif_path)
#                 if arr.ndim == 3 and arr.shape[2] >= 3:
#                     rgb16 = arr[..., [2, 1, 0]]  # R,G,B
#                 elif arr.ndim == 3 and arr.shape[0] >= 3:
#                     rgb16 = np.transpose(arr[[2,1,0], ...], (1,2,0))
#                 else:
#                     raise ValueError(f"波段数不足3: {arr.shape}")
#
#                 rgb8  = stretch_uint8(rgb16)
#                 Image.fromarray(rgb8).save(os.path.join(patch_dir, jpg_name), quality=95)
#                 print(f"[✓] patch_{patch_id} {ydir[-4:]}_{season} → {jpg_name}")
#             except Exception as e:
#                 print(f"[✗] {tif_path}: {e}")
#
# print("\n🎉 转换完成！请检查 datasets/rawframes/patch_xxx/ 目录下是否有 20 张 jpg")


# import rasterio
# import cv2
# import numpy as np
# from pathlib import Path
#
# input_tif = r"F:\zyp\数据文件夹\patch_308_summer.tif"
# output_jpg = r"F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes\patch_308\img_0002.jpg"
#
# with rasterio.open(input_tif) as src:
#     img = src.read([1, 2, 3])  # 读取 B4, B3, B2 波段 → RGB
#     img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
#     img = np.clip(img, 0, 3000) / 3000.0 * 255  # 线性拉伸到 0–255
#     img = img.astype(np.uint8)
#
# Path(output_jpg).parent.mkdir(parents=True, exist_ok=True)
# cv2.imwrite(output_jpg, img)

# import torch
#
# src_pth = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\videoswin\swin_tiny_patch244_window877_kinetics400_1k_converted.pth'
# dst_pth = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\pretrained\videoswin\swin_tiny_patch244_window877_kinetics400_1k_state_dict.pth'
#
# ckpt = torch.load(src_pth, map_location='cpu')
# if 'model' in ckpt:
#     state_dict = ckpt['model']
# else:
#     state_dict = ckpt
# torch.save(state_dict, dst_pth)
# print('Converted and saved:', dst_pth)


# import os
# import shutil
# from PIL import Image
# import numpy as np
#
# def is_black(img_path, threshold=10, percent=0.98):
#     img = Image.open(img_path).convert('L')  # 灰度
#     arr = np.array(img)
#     black_pixels = np.sum(arr < threshold)
#     total_pixels = arr.size
#     return (black_pixels / total_pixels) > percent
#
# def find_and_remove_black_patches(rawframes_dir, save_log=True):
#     patches = os.listdir(rawframes_dir)
#     removed_patches = []
#     for patch in patches:
#         patch_dir = os.path.join(rawframes_dir, patch)
#         if not os.path.isdir(patch_dir):
#             continue
#         imgs = [f for f in os.listdir(patch_dir) if f.endswith('.jpg')]
#         has_black = False
#         for img_name in imgs:
#             img_path = os.path.join(patch_dir, img_name)
#             try:
#                 if is_black(img_path):
#                     print(f"检测到黑色图片: {patch}/{img_name}")
#                     has_black = True
#                     break   # 一个就够，整个patch删
#             except Exception as e:
#                 print(f"读取失败: {img_path}, {e}")
#         if has_black:
#             try:
#                 shutil.rmtree(patch_dir)
#                 print(f"已删除patch: {patch}")
#                 removed_patches.append(patch)
#             except Exception as e:
#                 print(f"删除失败: {patch_dir}, {e}")
#     # 保存被删除的patch名单
#     if save_log and removed_patches:
#         log_path = os.path.join(rawframes_dir, "removed_patches.txt")
#         with open(log_path, 'w', encoding='utf-8') as f:
#             for patch in removed_patches:
#                 f.write(f"{patch}\n")
#         print(f"\n所有被删除的patch已保存到: {log_path}")
#
# if __name__ == "__main__":
#     rawframes_dir = "F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/rawframes"
#     find_and_remove_black_patches(rawframes_dir)

# import os
#
# def save_valid_patches(rawframes_dir, output_txt=None):
#     patches = [d for d in os.listdir(rawframes_dir) if os.path.isdir(os.path.join(rawframes_dir, d))]
#     patches.sort(key=lambda x: int(x.split('_')[1]))
#     if output_txt is None:
#         output_txt = os.path.join(rawframes_dir, 'remaining_patches.txt')
#     with open(output_txt, 'w') as f:
#         for patch in patches:
#             f.write(f"{patch}\n")
#     print(f"剩余有效 patch 数量：{len(patches)}，已保存到 {output_txt}")
#
# if __name__ == "__main__":
#     rawframes_dir = "F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/rawframes"
#     # 不传 output_txt 就默认保存在 rawframes 目录下
#     save_valid_patches(rawframes_dir)

def filter_split_file(split_txt, remain_txt, out_txt):
    # 读取 remain 列表
    with open(remain_txt, 'r') as f:
        remain_set = set(line.strip() for line in f if line.strip())
    # 过滤 split，每行第一个是 patch 名
    filtered = []
    with open(split_txt, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            patch_name = line.split()[0]
            if patch_name in remain_set:
                filtered.append(line)
    print(f"原始数量: {sum(1 for _ in open(split_txt))}，过滤后剩余: {len(filtered)}")
    # 保存
    with open(out_txt, 'w') as f:
        for line in filtered:
            f.write(line)

if __name__ == "__main__":
    # 修改为实际路径
    base = "F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/"
    remain_txt = base + "remaining_patches.txt"
    # val
    filter_split_file(base + "val.txt", remain_txt, base + "val_filtered.txt")
    # train
    filter_split_file(base + "train.txt", remain_txt, base + "train_filtered.txt")