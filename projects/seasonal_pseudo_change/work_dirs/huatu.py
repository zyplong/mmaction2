# # import re
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# #
# # # ====== 1. 路径配置 ======
# # log_path = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839/training.log'
# # pred_path = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839/pred.csv'
# # outdir = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/swin_tiny_raw_20250628_235839'
# #
# # # ====== 2. 解析训练日志并存 metrics.csv ======
# # rows, pat = [], re.compile(
# #     r'\[Epoch (\d+)] TrainLoss: ([\d.]+) \| SupLoss: ([\d.]+) '
# #     r'\| ConsisLoss: ([\d.]+) \| TrainAcc: ([\d.]+) \| ValAcc: ([\d.]+)'
# # )
# # with open(log_path, encoding='utf-8') as f:
# #     for line in f:
# #         m = pat.search(line)
# #         if m:
# #             rows.append([int(m.group(1))] + list(map(float, m.groups()[1:])))
# # df = pd.DataFrame(rows, columns=[
# #     'epoch','train_loss','sup_loss','consis_loss','train_acc','val_acc'])
# # df.to_csv(f'{outdir}/metrics.csv', index=False)
# #
# # # ====== 3. 画曲线图 curve_acc_loss.png ======
# # sns.set_style("whitegrid")
# # fig, ax1 = plt.subplots(figsize=(7,4))
# # ax1.plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
# # ax1.set_ylabel('Val Acc'); ax1.set_xlabel('Epoch'); ax1.set_ylim(0,1)
# # ax2 = ax1.twinx()
# # ax2.plot(df['epoch'], df['train_loss'], '--', label='Train Loss', color='C1')
# # ax2.set_ylabel('Train Loss')
# # fig.legend(loc='upper right')
# # fig.tight_layout()
# # fig.savefig(f'{outdir}/curve_acc_loss.png', dpi=300)
# # plt.close()
# #
# # # ====== 4. 混淆矩阵 confmat.png ======
# # pred_df = pd.read_csv(pred_path)
# # cm = confusion_matrix(pred_df.y_true, pred_df.y_pred, normalize='true')
# # disp = ConfusionMatrixDisplay(cm, display_labels=['No-Change','Change'])
# # disp.plot(cmap='Blues', values_format='.2f')
# # plt.tight_layout()
# # plt.savefig(f'{outdir}/confmat.png', dpi=300)
# # plt.close()
# #
# # # ====== 5. 最终结果表格 Table1_baseline.md ======
# # final = df.iloc[-1]
# # table_md = (
# #     f"| Clip | Semi | ValAcc | TrainAcc | SupLoss | ConsisLoss |\n"
# #     f"|------|------|--------|----------|---------|------------|\n"
# #     f"| 20×RGB | ✓ | **{final.val_acc:.3f}** | {final.train_acc:.3f} | "
# #     f"{final.sup_loss:.3f} | {final.consis_loss:.3f} |"
# # )
# # with open(f'{outdir}/Table1_baseline.md','w',encoding='utf-8') as f:
# #     f.write(table_md)
# # print(table_md)
# #
# # print('✅ 所有可视化文件已生成，直接用于论文和汇报！')
#
#
# # import os
# # from PIL import Image, ImageDraw, ImageFont
# # import pandas as pd
# #
# # # 配置
# # rawframes_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
# # pred_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\pred.csv'
# # out_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs\swin_tiny_raw_20250628_235839\vis_patch'
# # os.makedirs(out_dir, exist_ok=True)
# #
# # # 加载结果
# # df = pd.read_csv(pred_csv)
# # # 随机挑几个正确/错误/特殊样本
# # samples = {
# #     'TP': df[(df.y_true == 1) & (df.y_pred == 1)].sample(n=min(2, len(df[(df.y_true == 1) & (df.y_pred == 1)])), random_state=1),
# #     'TN': df[(df.y_true == 0) & (df.y_pred == 0)].sample(n=min(2, len(df[(df.y_true == 0) & (df.y_pred == 0)])), random_state=2),
# #     'FP': df[(df.y_true == 0) & (df.y_pred == 1)].sample(n=min(2, len(df[(df.y_true == 0) & (df.y_pred == 1)])), random_state=3),
# #     'FN': df[(df.y_true == 1) & (df.y_pred == 0)].sample(n=min(2, len(df[(df.y_true == 1) & (df.y_pred == 0)])), random_state=4),
# # }
# #
# # for typ, rows in samples.items():
# #     for i, row in rows.iterrows():
# #         patch_id = row['patch_id']
# #         imgs = []
# #         for j in [1, 10, 20]:  # 只取第1/10/20帧
# #             img_path = os.path.join(rawframes_dir, patch_id, f'img_{j:04d}.jpg')
# #             img = Image.open(img_path).resize((224, 224))
# #             # 指定更大字号
# #             try:
# #                 font = ImageFont.truetype("arial.ttf", 28)
# #             except:
# #                 font = None
# #             draw = ImageDraw.Draw(img)
# #             # 用白色描边
# #             draw.text(
# #                 (5, 5), f"{patch_id}\nGT:{row['y_true']} Pred:{row['y_pred']}",
# #                 fill='red', font=font, stroke_width=2, stroke_fill='white'
# #             )
# #             imgs.append(img)
# #         # 拼接
# #         out_img = Image.new('RGB', (224 * 3, 224))
# #         for k, img in enumerate(imgs):
# #             out_img.paste(img, (224 * k, 0))
# #         out_img.save(os.path.join(out_dir, f"{patch_id}_{typ}.jpg"))
#
#
# # import os
# # import shutil
# # import numpy as np
# # from PIL import Image
# # import tifffile as tiff
# #
# # src_rgb_root = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
# # src_ndvi_root = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\all_ndvi'
# # dst_root = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes_ndvi'
# #
# # SEASONS = ['spring', 'summer', 'autumn', 'winter']
# # YEARS = [2019, 2020, 2021, 2022, 2023]
# # NDVI_LIST = [f"{year}_{season}_NDVI.tif" for year in YEARS for season in SEASONS]  # 5*4=20
# #
# # for patch in os.listdir(src_rgb_root):
# #     rgb_dir = os.path.join(src_rgb_root, patch)
# #     ndvi_dir = os.path.join(src_ndvi_root, patch)
# #     if not (os.path.isdir(rgb_dir) and os.path.isdir(ndvi_dir)):
# #         print(f"跳过 {patch}（不是文件夹或没有对应 NDVI）")
# #         continue
# #
# #     dst_dir = os.path.join(dst_root, patch)
# #     os.makedirs(dst_dir, exist_ok=True)
# #
# #     rgb_files = [f"img_{i+1:04d}.jpg" for i in range(20)]
# #     ndvi_files = NDVI_LIST
# #
# #     ndvi_missing = [f for f in ndvi_files if not os.path.exists(os.path.join(ndvi_dir, f))]
# #     if ndvi_missing:
# #         print(f"{patch} 缺失NDVI: {ndvi_missing}，已跳过")
# #         continue
# #
# #     # 复制RGB
# #     for rgb_name in rgb_files:
# #         rgb_src_path = os.path.join(rgb_dir, rgb_name)
# #         if not os.path.exists(rgb_src_path):
# #             print(f"{patch} 缺失RGB: {rgb_name}，已跳过")
# #             continue
# #         shutil.copy(rgb_src_path, os.path.join(dst_dir, rgb_name))
# #
# #     # NDVI转换并重命名
# #     for i in range(20):
# #         ndvi_src_path = os.path.join(ndvi_dir, ndvi_files[i])
# #         rgb_basename = rgb_files[i][:-4]
# #         ndvi_dst_name = f"{rgb_basename}_ndvi.jpg"
# #         ndvi_dst_path = os.path.join(dst_dir, ndvi_dst_name)
# #         try:
# #             arr = tiff.imread(ndvi_src_path)
# #             # 归一化到8bit
# #             if arr.dtype != np.uint8:
# #                 arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype('uint8')
# #             img = Image.fromarray(arr)
# #             img = img.convert('L')
# #             img.save(ndvi_dst_path, quality=95)
# #         except Exception as e:
# #             print(f"{ndvi_src_path} 读取或保存失败：", e)
# #     print(f"{patch} 合并完毕！")
# #
# # print("所有patch已合并完毕！")
#
#
# import os
# from glob import glob
#
# # 根目录
# rawframes_root = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\rawframes'
# ndvi_root      = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\all_ndvi'
#
# patch_dirs = sorted(os.listdir(rawframes_root))  # 所有 patch_xx 文件夹
#
# for patch in patch_dirs:
#     rgb_dir = os.path.join(rawframes_root, patch)
#     ndvi_dir = os.path.join(ndvi_root, patch)
#     if not os.path.exists(ndvi_dir):
#         print(f"跳过不存在的 NDVI 目录: {ndvi_dir}")
#         continue
#
#     # 按文件名排序获取 RGB 和 NDVI 文件名
#     rgb_imgs = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
#     ndvi_imgs = sorted([f for f in os.listdir(ndvi_dir) if f.endswith('.tif')])
#
#     if len(rgb_imgs) != len(ndvi_imgs):
#         print(f"[警告] {patch} 下 RGB/NDVI 数量不一致：{len(rgb_imgs)} vs {len(ndvi_imgs)}，已跳过！")
#         continue
#
#     # 开始重命名
#     for rgb_name, ndvi_old_name in zip(rgb_imgs, ndvi_imgs):
#         ndvi_new_name = rgb_name.replace('.jpg', '.tif')
#         src = os.path.join(ndvi_dir, ndvi_old_name)
#         dst = os.path.join(ndvi_dir, ndvi_new_name)
#         if src != dst:
#             print(f"{ndvi_old_name} -> {ndvi_new_name}")
#             os.rename(src, dst)
#
# print("全部 NDVI 文件已重命名完毕！")


import os
from PIL import Image
import pandas as pd

# ———— 1. 把这里改成你自己的 unlabeled.csv（或 .txt）绝对路径 —————
UNLABELED_CSV = r"F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\unlabeled.txt"

# ———— 2. 把这里改成你 NDVI 图像所在的根目录 ————————————
# 目录结构假设为 NDVI_ROOT/<pid>/img_0001.tif ... img_0008.tif
NDVI_ROOT     = r"F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/data/ndvi"

# 读取 “无标签” 标注文件，取出所有 pid
ann   = pd.read_csv(
    UNLABELED_CSV,
    sep=r'\s+',
    header=None,
    names=['pid','fid','label'],
    engine='python'
)
pids  = ann['pid'].unique().tolist()

clip_len = 8  # 和你训练脚本里用的一致

bad = []
for pid in pids:
    for i in range(1, clip_len+1):
        tif_path = os.path.join(NDVI_ROOT, str(pid), f"img_{i:04d}.tif")
        try:
            # verify() 只检测文件完整性，不解码到内存
            with Image.open(tif_path) as im:
                im.verify()
        except Exception as e:
            bad.append((pid, i, tif_path, str(e)))

if bad:
    print("以下文件损坏或无法打开：")
    for pid, idx, path, err in bad:
        print(f"  pid={pid}, frame={idx:04d} → {path}\n    错误：{err}")
else:
    print("全部 NDVI TIFF 文件都能正常打开 ✔")