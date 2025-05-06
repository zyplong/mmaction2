"""
auto_label_mapbiomas.py
--------------------------------------------------------
按 patch-ID 的中心像元，比较 2019 vs 2023
MapBiomas = 3 (cultivated area) → 判断真假变化
--------------------------------------------------------
"""

import os, csv
import rasterio

# ★ 改成你的 data 绝对路径 ----------------------------
DATA_ROOT = r"E:\paper\mmaction2\projects\seasonal_pseudo_change\downstream\data"
# -----------------------------------------------------

PATCH_DIR = os.path.join(DATA_ROOT, "rawframes")          # patch_x 目录
MAP19     = os.path.join(DATA_ROOT, "brasil_coverage_2019.tif")
MAP23     = os.path.join(DATA_ROOT, "brasil_coverage_2023.tif")
OUT_CSV   = os.path.join(DATA_ROOT, "labels_auto_generated.csv")

def sample_class(raster, lon, lat):
    """在 (lon,lat) 采样第 1 波段的整型值"""
    with rasterio.open(raster) as src:
        return int(next(src.sample([(lon, lat)]))[0])

def center_lonlat(tif_path):
    """返回一个 GeoTIFF 的中心经纬度"""
    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
    return (left + right) / 2, (bottom + top) / 2

# 找到所有 patch 目录
patches = [d for d in os.listdir(PATCH_DIR) if d.startswith("patch_")]
patches.sort(key=lambda x: int(x.split("_")[1]))          # 按编号排序 (可选)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "label"])

    for p in patches:
        # 直接用 2019 里的 spring 影像求中心坐标
        tif_2019 = os.path.join(DATA_ROOT, "gee_farmland_2019", f"{p}_spring.tif")
        if not os.path.isfile(tif_2019):
            print("× 找不到", tif_2019, "跳过")
            continue

        lon, lat = center_lonlat(tif_2019)

        cls19 = sample_class(MAP19, lon, lat)
        cls23 = sample_class(MAP23, lon, lat)

        if (cls19 == 3) and (cls23 != 3):
            label = "real_change"      # 真实变化
        elif (cls19 == 3) and (cls23 == 3):
            label = "pseudo_change"    # 伪变化
        else:
            label = "unknown"

        writer.writerow([p, label])
        print(f"{p:8s}  -> {label}")

print("\n✔  自动标签已写到", OUT_CSV)
