"""
organise_patches_s2.py
----------------------
把 2019-2023 Sentinel-2 4-band GeoTIFF
转成按时间顺序连续编号的 jpg，供 SeCo / Video Swin 使用
"""

import os, re, numpy as np, tifffile
from PIL import Image

# ========= 1. 路径 =========
PROJ_ROOT = r"F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change"
SRC_ROOT  = os.path.join(PROJ_ROOT, "datasets", "300")         # gee_farmland_20YY
DST_ROOT  = os.path.join(PROJ_ROOT, "datasets", "rawframes")   # 输出 jpg
os.makedirs(DST_ROOT, exist_ok=True)

# ========= 2. 文件名解析 =========
pat = re.compile(r"patch_(\d+)_(spring|summer|autumn|winter)\.tif$", re.I)
season_order = ['spring', 'summer', 'autumn', 'winter']

# 年份文件夹按年份排序
year_dirs = sorted(
    [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))],
    key=lambda d: d.split('_')[-1]                             # 2019→2020→…
)
year2idx = {d: i for i, d in enumerate(year_dirs)}

# ========= 3. 百分位拉伸函数 =========
def stretch_uint8(arr16, p_low=1, p_high=99):
    """把 0-10000 的 uint16 拉伸到 0-255"""
    arr = arr16.astype(np.float32)
    lo, hi = np.percentile(arr, [p_low, p_high])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)

# ========= 4. 主循环 =========
for ydir in year_dirs:                                 # 逐年
    yidx = year2idx[ydir]
    ypath = os.path.join(SRC_ROOT, ydir)

    for fname in os.listdir(ypath):
        m = pat.match(fname)
        if not m:
            continue

        patch_id = int(m.group(1))
        season   = m.group(2).lower()
        sidx     = season_order.index(season)

        # 连续全局帧编号：0…15(或19)
        gidx   = yidx * 4 + sidx
        jpg_nm = f"img_{gidx+1:04d}.jpg"

        patch_dir = os.path.join(DST_ROOT, f"patch_{patch_id}")
        os.makedirs(patch_dir, exist_ok=True)

        tif_path = os.path.join(ypath, fname)
        try:
            arr = tifffile.imread(tif_path)           # [H,W,4]  int16
            if arr.shape[2] < 4:
                raise ValueError(f"波段数不足 4：{arr.shape}")

            # Sentinel-2 : B4, B3, B2 → R, G, B
            rgb16 = arr[..., [2, 1, 0]]               # 若实为 B3,B2,B1,B4，请改索引
            rgb8  = stretch_uint8(rgb16)

            Image.fromarray(rgb8).save(os.path.join(patch_dir, jpg_nm), quality=95)
            print(f"[✓] patch_{patch_id:03d} {ydir[-4:]}_{season} → {jpg_nm}")

            # ---------- (可选) 保存 NDVI -----------
            # nir = arr[..., 3].astype(np.float32)
            # red = arr[..., 2].astype(np.float32)
            # ndvi = (nir - red) / (nir + red + 1e-6)
            # np.save(os.path.join(patch_dir, f"ndvi_{gidx+1:04d}.npy"), ndvi)

        except Exception as e:
            print(f"[✗] {tif_path}: {e}")

print("\n🎉 整理完成！请到 datasets/rawframes 查看每个 patch 是否有 16/20 张彩色 jpg。")



# import os, numpy as np
# import tifffile
# from PIL import Image
#
# # === 路径 ===
# src_root = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\300'
# dst_rgb  = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
# dst_ndvi = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\ndvi_arrays'
# os.makedirs(dst_rgb,  exist_ok=True)
# os.makedirs(dst_ndvi, exist_ok=True)
#
# season_order = ['spring', 'summer', 'autumn', 'winter']
# patch_dict   = {}
#
# # === 1. 收集文件路径 ===
# for year_dir in sorted(os.listdir(src_root)):
#     year_path = os.path.join(src_root, year_dir)
#     if not os.path.isdir(year_path):  continue
#     year = year_dir.split('_')[-1]    # 2019 ...
#     for fname in os.listdir(year_path):
#         if not fname.endswith('.tif'):  continue
#         parts = fname.replace('.tif','').split('_')  # patch_27_autumn.tif
#         if len(parts) != 3: continue
#         patch_id = int(parts[1])
#         season   = parts[2]
#         patch_dict.setdefault(patch_id, {})[season] = os.path.join(year_path, fname)
#
# # === 2. 转换 ===
# for patch_id, season_files in patch_dict.items():
#     rgb_dir  = os.path.join(dst_rgb,  f'patch_{patch_id}')
#     ndvi_dir = os.path.join(dst_ndvi, f'patch_{patch_id}')
#     os.makedirs(rgb_dir,  exist_ok=True)
#     os.makedirs(ndvi_dir, exist_ok=True)
#
#     for idx, season in enumerate(season_order):
#         if season not in season_files: continue
#         tif_path = season_files[season]
#
#         try:
#             img = tifffile.imread(tif_path)         # shape [H,W,4]  (R,G,B,NDVI)
#             if img.ndim != 3 or img.shape[2] < 4:
#                 raise ValueError(f"expect 4-channel, got {img.shape}")
#             rgb  = np.clip(img[:, :, :3], 0, 255).astype(np.uint8)   # R,G,B
#             ndvi = img[:, :, 3]  # 第4通道  (可根据数据实际调整)
#
#             # --- 保存 RGB ---
#             jpg_name = f'img_{idx+1:04d}.jpg'
#             Image.fromarray(rgb).save(os.path.join(rgb_dir, jpg_name), quality=95)
#
#             # --- 保存 NDVI ---
#             np.save(os.path.join(ndvi_dir, f'ndvi_{idx+1:04d}.npy'), ndvi)
#
#             print(f"[✓] patch_{patch_id:03d} {season} done")
#
#         except Exception as e:
#             print(f"[✗] skip {tif_path}: {e}")
#
# print("🎉  RGB .jpg + NDVI .npy 整理完成！")
