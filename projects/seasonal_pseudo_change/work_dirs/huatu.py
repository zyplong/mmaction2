# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
# # ====== 1. 路径配置 ======
# log_path = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839/training.log'
# pred_path = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839/pred.csv'
# outdir = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839'
#
# # ====== 2. 解析训练日志并存 metrics.csv ======
# rows, pat = [], re.compile(
#     r'\[Epoch (\d+)] TrainLoss: ([\d.]+) \| SupLoss: ([\d.]+) '
#     r'\| ConsisLoss: ([\d.]+) \| TrainAcc: ([\d.]+) \| ValAcc: ([\d.]+)'
# )
# with open(log_path, encoding='utf-8') as f:
#     for line in f:
#         m = pat.search(line)
#         if m:
#             rows.append([int(m.group(1))] + list(map(float, m.groups()[1:])))
# df = pd.DataFrame(rows, columns=[
#     'epoch','train_loss','sup_loss','consis_loss','train_acc','val_acc'])
# df.to_csv(f'{outdir}/metrics.csv', index=False)
#
# # ====== 3. 画曲线图 curve_acc_loss.png ======
# sns.set_style("whitegrid")
# fig, ax1 = plt.subplots(figsize=(7,4))
# ax1.plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
# ax1.set_ylabel('Val Acc'); ax1.set_xlabel('Epoch'); ax1.set_ylim(0,1)
# ax2 = ax1.twinx()
# ax2.plot(df['epoch'], df['train_loss'], '--', label='Train Loss', color='C1')
# ax2.set_ylabel('Train Loss')
# fig.legend(loc='upper right')
# fig.tight_layout()
# fig.savefig(f'{outdir}/curve_acc_loss.png', dpi=300)
# plt.close()
#
# # ====== 4. 混淆矩阵 confmat.png ======
# pred_df = pd.read_csv(pred_path)
# cm = confusion_matrix(pred_df.y_true, pred_df.y_pred, normalize='true')
# disp = ConfusionMatrixDisplay(cm, display_labels=['No-Change','Change'])
# disp.plot(cmap='Blues', values_format='.2f')
# plt.tight_layout()
# plt.savefig(f'{outdir}/confmat.png', dpi=300)
# plt.close()
#
# # ====== 5. 最终结果表格 Table1_baseline.md ======
# final = df.iloc[-1]
# table_md = (
#     f"| Clip | Semi | ValAcc | TrainAcc | SupLoss | ConsisLoss |\n"
#     f"|------|------|--------|----------|---------|------------|\n"
#     f"| 20×RGB | ✓ | **{final.val_acc:.3f}** | {final.train_acc:.3f} | "
#     f"{final.sup_loss:.3f} | {final.consis_loss:.3f} |"
# )
# with open(f'{outdir}/Table1_baseline.md','w',encoding='utf-8') as f:
#     f.write(table_md)
# print(table_md)
#
# print('✅ 所有可视化文件已生成，直接用于论文和汇报！')


import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# 配置
rawframes_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
pred_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\pred.csv'
out_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\vis_patch'
os.makedirs(out_dir, exist_ok=True)

# 加载结果
df = pd.read_csv(pred_csv)
# 随机挑几个正确/错误/特殊样本
samples = {
    'TP': df[(df.y_true == 1) & (df.y_pred == 1)].sample(n=min(2, len(df[(df.y_true == 1) & (df.y_pred == 1)])), random_state=1),
    'TN': df[(df.y_true == 0) & (df.y_pred == 0)].sample(n=min(2, len(df[(df.y_true == 0) & (df.y_pred == 0)])), random_state=2),
    'FP': df[(df.y_true == 0) & (df.y_pred == 1)].sample(n=min(2, len(df[(df.y_true == 0) & (df.y_pred == 1)])), random_state=3),
    'FN': df[(df.y_true == 1) & (df.y_pred == 0)].sample(n=min(2, len(df[(df.y_true == 1) & (df.y_pred == 0)])), random_state=4),
}

for typ, rows in samples.items():
    for i, row in rows.iterrows():
        patch_id = row['patch_id']
        imgs = []
        for j in [1, 10, 20]:  # 只取第1/10/20帧
            img_path = os.path.join(rawframes_dir, patch_id, f'img_{j:04d}.jpg')
            img = Image.open(img_path).resize((224, 224))
            # 指定更大字号
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = None
            draw = ImageDraw.Draw(img)
            # 用白色描边
            draw.text(
                (5, 5), f"{patch_id}\nGT:{row['y_true']} Pred:{row['y_pred']}",
                fill='red', font=font, stroke_width=2, stroke_fill='white'
            )
            imgs.append(img)
        # 拼接
        out_img = Image.new('RGB', (224 * 3, 224))
        for k, img in enumerate(imgs):
            out_img.paste(img, (224 * k, 0))
        out_img.save(os.path.join(out_dir, f"{patch_id}_{typ}.jpg"))