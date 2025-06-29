
#txt去除， 换成空格
# import glob
# import os
#
# files = [
#     r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\train.txt',
#     r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\val.txt'
# ]
#
# for fname in files:
#     with open(fname, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     with open(fname, 'w', encoding='utf-8') as f:
#         for line in lines:
#             # 把逗号替换成空格，再去掉首尾空白
#             newline = line.replace(',', ' ').strip()
#             # 去除多余的空格
#             newline = ' '.join(newline.split())
#             f.write(newline + '\n')
#     print(f"{fname} 已完成替换")

# import os
#
# rawframes_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
#
# for patch_name in os.listdir(rawframes_dir):
#     patch_dir = os.path.join(rawframes_dir, patch_name)
#     if not os.path.isdir(patch_dir):
#         continue
#     # 只处理 .jpg 文件
#     imgs = sorted([f for f in os.listdir(patch_dir) if f.endswith('.jpg')])
#     for idx, old_name in enumerate(imgs):
#         new_name = f'img_{idx+1:04d}.jpg'
#         old_path = os.path.join(patch_dir, old_name)
#         new_path = os.path.join(patch_dir, new_name)
#         os.rename(old_path, new_path)
#     print(f'{patch_name}: renamed {len(imgs)} images')


import re, argparse, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True)
parser.add_argument('--pred', required=True)
parser.add_argument('--outdir', default='results')
args = parser.parse_args()
sns.set_style("whitegrid")               # 论文友好配色

# 1. 解析 training_log.txt ➜ metrics.csv
rows, pat = [], re.compile(
    r'\[Epoch (\d+)] TrainLoss: ([\d.]+) \| SupLoss: ([\d.]+) '
    r'\| ConsisLoss: ([\d.]+) \| TrainAcc: ([\d.]+) \| ValAcc: ([\d.]+)'
)
with open(args.log, encoding='utf-8') as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append([int(m.group(1))] + list(map(float, m.groups()[1:])))
df = pd.DataFrame(rows, columns=[
    'epoch','train_loss','sup_loss','consis_loss','train_acc','val_acc'])
df.to_csv(f'{args.outdir}/metrics.csv', index=False)

# 2. 曲线：ValAcc & TrainLoss
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
ax1.set_ylabel('Val Acc'); ax1.set_xlabel('Epoch'); ax1.set_ylim(0,1)
ax2 = ax1.twinx()
ax2.plot(df['epoch'], df['train_loss'], '--', label='Train Loss', color='C1')
ax2.set_ylabel('Train Loss')
fig.legend(loc='upper right'); fig.tight_layout()
fig.savefig(f'{args.outdir}/curve_acc_loss.png', dpi=300)

# 3. 混淆矩阵（2 类示例）
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pred_df = pd.read_csv(args.pred)     # columns: patch_id, y_true, y_pred
cm = confusion_matrix(pred_df.y_true, pred_df.y_pred, normalize='true')
disp = ConfusionMatrixDisplay(cm, display_labels=['No-Change','Change'])
disp.plot(cmap='Blues', values_format='.2f')
plt.tight_layout()
plt.savefig(f'{args.outdir}/confmat.png', dpi=300)

# 4. 生成 LaTeX/Markdown 表格（最后一行指标）
final = df.iloc[-1]
table_md = (
f"| Clip | Semi | ValAcc | TrainAcc | SupLoss | ConsisLoss |\n"
f"|------|------|--------|----------|---------|------------|\n"
f"| 20×RGB | ✓ | **{final.val_acc:.3f}** | {final.train_acc:.3f} | "
f"{final.sup_loss:.3f} | {final.consis_loss:.3f} |"
)
open(f'{args.outdir}/Table1_baseline.md','w').write(table_md)
print(table_md)