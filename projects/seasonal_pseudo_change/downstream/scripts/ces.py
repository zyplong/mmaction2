# merge_labels.py  ―― 直接复制保存，然后 python merge_labels.py
import pandas as pd, os, pathlib

ROOT = pathlib.Path(r"E:\paper\mmaction2\projects\seasonal_pseudo_change\downstream\data")
auto  = pd.read_csv(ROOT / "labels_auto_generated.csv")        # 300 行
manual = pd.read_csv(ROOT / "todo_manual.csv")                 # ≈120 行

# 以人工标注为准，覆盖自动结果
merged = auto.set_index("video_path")
merged.update(manual.set_index("video_path"))          # 只有 label 列会被替换
merged = merged.reset_index()

merged.to_csv(ROOT / "labels_final.csv", index=False)
print("✅  已生成 labels_final.csv  →", merged["label"].value_counts())
