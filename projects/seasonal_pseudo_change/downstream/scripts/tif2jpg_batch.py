"""
一次性：
gee_farmland_2019 ~ gee_farmland_2023
    └─ patch_1_spring.tif ...
→ data/rawframes/
    ├─ patch_1/
    │   img_00001.jpg  ...  img_00020.jgw
    ├─ patch_2/
    └─ ...
"""
import os, re, rasterio
import numpy as np
from PIL import Image

PROJ_ROOT  = r"E:\paper\mmaction2\projects\seasonal_pseudo_change"
DATASET    = os.path.join(PROJ_ROOT, "downstream", "data")
OUT_ROOT   = os.path.join(PROJ_ROOT, "downstream", "data", "rawframes")
os.makedirs(OUT_ROOT, exist_ok=True)

# 正则抽 patchID & season
pat = re.compile(r"patch_(\d+)_(spring|summer|autumn|winter)\.tif$", re.I)

def tif2jpg(tif_path, out_jpg):
    with rasterio.open(tif_path) as src:
        rgb = src.read([4,3,2]).astype(np.float32)      # 4-3-2 → RGB
        rgb = np.clip((rgb/3000)*255, 0, 255).astype(np.uint8)
        Image.fromarray(np.transpose(rgb, (1,2,0))).save(out_jpg, quality=90)

        # world file  (*.jgw)——QGIS 还能直接叠加
        with open(out_jpg.replace(".jpg", ".jgw"), "w") as f:
            f.write("\n".join([f"{src.transform.a:.10f}",
                               "0.0000000000",
                               "0.0000000000",
                               f"{-src.transform.e:.10f}",
                               f"{src.transform.c:.10f}",
                               f"{src.transform.f:.10f}"]))

count = 0
for year in os.listdir(DATASET):                                   # 5 个年份文件夹
    year_dir = os.path.join(DATASET, year)
    if not os.path.isdir(year_dir): continue

    for tif in os.listdir(year_dir):
        m = pat.search(tif)
        if not m: continue
        patch_id, season = m.groups()                              # ① patch号 ②季节

        patch_dir = os.path.join(OUT_ROOT, f"patch_{patch_id}")
        os.makedirs(patch_dir, exist_ok=True)

        # 每个 patch 下统一 20 帧，按 seasons 固定顺序写序号
        idx_map = dict(spring = 1, summer = 2, autumn = 3, winter = 4)
        start = (idx_map[season]-1)*4 + 1                          # 1,5,9,13
        for i in range(4):                                         # 4 张 → 4 帧
            tif_path = os.path.join(year_dir, tif)
            out_name = f"img_{start+i:05d}.jpg"
            out_jpg  = os.path.join(patch_dir, out_name)
            tif2jpg(tif_path, out_jpg)
        count += 4

print(f"✅  已输出 {count} 张 JPG →  {OUT_ROOT}")
