# import os
# import numpy as np
# import pandas as pd
#
# # 配置路径
# unlabeled_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\unlabeled.csv'
# logits_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\results\logits'
# output_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\pseudo_label.csv'
#
# # 1. 读取unlabeled patch名
# patches = pd.read_csv(unlabeled_csv, header=None)[0].tolist()
#
# # 2. 检查npy是否存在，并生成标签
# records = []
# not_found = []
# for patch in patches:
#     npy_path = os.path.join(logits_dir, f"{patch}.npy")
#     if os.path.exists(npy_path):
#         logits = np.load(npy_path)
#         # 取最大概率的类别，支持多种形状
#         if logits.ndim == 1:
#             label = int(np.argmax(logits))
#         elif logits.ndim >= 2:
#             # 有可能是 [N, num_class]，先取最后一帧平均
#             label = int(np.argmax(logits.mean(axis=0)))
#         else:
#             label = -1  # 异常
#         records.append([patch, label])
#     else:
#         not_found.append(patch)
#         records.append([patch, -1])  # 或者直接不写入
#
# # 3. 写csv
# df = pd.DataFrame(records, columns=['patch', 'pseudo_label'])
# df.to_csv(output_csv, index=False)
# print(f"Done! 写入{len(records)}条, 缺失npy文件{len(not_found)}条：", not_found)


import os
import numpy as np
import pandas as pd

# --------- 配置路径 ---------
logits_dir = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\results\logits'  # 你的 logits .npy 文件夹
unlabeled_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\unlabeled.csv'  # unlabeled patch列表
output_csv = r'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\datasets\splits\pseudo_label.csv'  # 输出伪标签csv

# --------- 读取 unlabeled patch 列表 ---------
patches = pd.read_csv(unlabeled_csv, header=None)[0].tolist()  # 一列patch名

records = []
for patch in patches:
    npy_path = os.path.join(logits_dir, f"{patch}.npy")
    if not os.path.exists(npy_path):
        print(f"Warning: {npy_path} not found, skip.")
        continue
    logits = np.load(npy_path)
    # 支持 shape (num_classes) 或 (1, num_classes)
    if logits.ndim > 1:
        logits = logits.reshape(-1)
    label = int(np.argmax(logits))
    records.append([patch, label])

# --------- 写入 csv ---------
df = pd.DataFrame(records, columns=['patch', 'pseudo_label'])
df.to_csv(output_csv, index=False)
print(f"已保存到: {output_csv}")