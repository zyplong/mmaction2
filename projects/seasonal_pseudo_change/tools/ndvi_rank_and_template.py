# """
# ndvi_rank_and_template.py
# -------------------------
# 按照 1300 个 .tif / .npy 的实际目录结构计算 NDVI_range
# 并把结果、空白标注表各写一份 CSV
# """
#
# import os, re, glob, csv
# import numpy as np
# import tifffile
#
# # ========= 1. 路径 =========
# ROOT_DIR   = r"F:\zyp\数据文件夹\1000"             # 你的 gee_farmland_20YY 所在根
# OUT_RANK   = r"F:\zyp\数据文件夹\ndvi_range_ranking.csv"
# OUT_LABELS = r"F:\zyp\数据文件夹\label_template.csv"
#
# SKIP_BEFORE = 300          # patch_001 – patch_300 已有人类标注 → 跳过
# TIF_RE      = re.compile(r"patch_(\d+)_(spring|summer|autumn|winter)\.tif$", re.I)
# NPY_RE      = re.compile(r"ndvi_(\d+)\.npy$", re.I)             # 可选
#
# # ========= 2. 读 NDVI 的小工具 =========
# def ndvi_from_tif(path:str) -> np.ndarray:
#     arr = tifffile.imread(path)                               # (H,W,4) uint16
#     nir, red = arr[..., 3], arr[..., 2]                       # Sentinel-2: B8, B4
#     return (nir.astype(np.float32) - red) / (nir + red + 1e-6)
#
# def ndvi_from_npy(path:str) -> np.ndarray:
#     return np.load(path)                                      # (H,W) float32
#
# # ========= 3. 先把所有文件按 patch_id 收集到字典 =========
# patch_frames = {}                                             # {id: [ndvi_1, …]}
# for year_dir in sorted(glob.glob(os.path.join(ROOT_DIR, "gee_farmland_*"))):
#     for fname in os.listdir(year_dir):
#         fpath = os.path.join(year_dir, fname)
#
#         # ----- .tif -----
#         m = TIF_RE.match(fname)
#         if m:
#             pid = int(m.group(1))
#             if pid <= SKIP_BEFORE:  continue
#             patch_frames.setdefault(pid, []).append(ndvi_from_tif(fpath))
#             continue
#
#         # ----- .npy (可选，有就直接堆进去) -----
#         m = NPY_RE.match(fname)
#         if m:
#             pid = int(m.group(1))
#             if pid <= SKIP_BEFORE:  continue
#             patch_frames.setdefault(pid, []).append(ndvi_from_npy(fpath))
#             continue
#
# # ========= 4. 计算 NDVI_range =========
# rank_rows = []
# for pid, frames in patch_frames.items():
#     stack = np.stack(frames, axis=0)          # (T,H,W)
#     ndvi_range = float(stack.max() - stack.min())
#     rank_rows.append((pid, ndvi_range))
#
# # 按降序排序
# rank_rows.sort(key=lambda x: x[1], reverse=True)
#
# # ========= 5. 写 CSV =========
# with open(OUT_RANK, "w", newline="") as f:
#     csv.writer(f).writerows([("patch_id", "ndvi_range")] + rank_rows)
#
# with open(OUT_LABELS, "w", newline="") as f:
#     csv.writer(f).writerows([("patch_id", "label")] + [(pid, "") for pid, _ in rank_rows])
#
# print(f"🎉 排序结果写入: {OUT_RANK}")
# print(f"📝 空白标注表写入: {OUT_LABELS}")
# print(f"   共统计到 {len(rank_rows)} 个 patch（从 301 开始）。")

"""
rename_patch_ids.py
-------------------
把 patch_1~patch_1000 重命名为 patch_301~patch_1300，
并重新组织为每年一个文件夹：gee_farmland_2019 ~ gee_farmland_2023。
"""

"""
filter_patch_301_to_1300.py
----------------------------
从原始 5 年影像中筛选 patch_301 ~ patch_1300，
不改文件名，复制到新的 gee_farmland_20XX 文件夹中。
"""

import os
import shutil
import re

# 原始数据目录
SRC_ROOT = r"F:\zyp\数据文件夹\1000"

# 输出新结构目录
DST_ROOT = r"F:\zyp\数据文件夹\1000_filtered"
os.makedirs(DST_ROOT, exist_ok=True)

# patch_id 范围
KEEP_MIN = 301
KEEP_MAX = 1300

# 年份目录
years = ['2019', '2020', '2021', '2022', '2023']
pattern = re.compile(r'patch_(\d+)_\w+\.tif$', re.I)

for year in years:
    src_year_dir = os.path.join(SRC_ROOT, f"gee_farmland_{year}")
    dst_year_dir = os.path.join(DST_ROOT, f"gee_farmland_{year}")
    os.makedirs(dst_year_dir, exist_ok=True)

    for fname in os.listdir(src_year_dir):
        m = pattern.match(fname)
        if not m:
            continue

        patch_id = int(m.group(1))
        if KEEP_MIN <= patch_id <= KEEP_MAX:
            src_path = os.path.join(src_year_dir, fname)
            dst_path = os.path.join(dst_year_dir, fname)
            shutil.copy2(src_path, dst_path)
            print(f"[✓] {fname} copied to {year}/")

print("\n🎉 筛选完成：patch_301 ~ patch_1300 已复制到新目录。")