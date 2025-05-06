import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(r"E:\paper\mmaction2\projects\seasonal_pseudo_change\downstream\data")
CSV  = ROOT / "labels_final.csv"
OUT  = ROOT / "splits"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

# 只保留 已知标签
df = df[ df["label"].isin(["pseudo_change", "real_change"]) ].copy()

# 映射到数字
label_map = {"pseudo_change":0, "real_change":1}
df["num"] = df["label"].map(label_map)

# 写 label_map.txt 方便以后 config 里读取
with open(ROOT/"label_map.txt","w",encoding="utf-8") as f:
    f.write("0 pseudo_change\n1 real_change\n")

# 80/20 随机划分（可调整 random_state）
train, val = train_test_split(df, test_size=0.2, stratify=df["num"],
                              random_state=42)

def dump(split, name):
    out = OUT / f"{name}.csv"
    split_out = pd.DataFrame({
        "video_path": split["video_path"],
        "total_frames": 4,          # 你每个 patch 正好四季 4 帧
        "label": split["num"]
    })
    split_out.to_csv(out, index=False)
    print(f"✅  写出 {out.relative_to(ROOT.parent)}   ({len(split_out)} 条)")

dump(train, "train")
dump(val,   "val")
