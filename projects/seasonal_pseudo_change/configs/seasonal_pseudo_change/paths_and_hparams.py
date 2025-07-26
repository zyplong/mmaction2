# configs/seasonal_pseudo_change/paths_and_hparams.py
# ——— 1. 数据划分文件 ———
ann_file_train     = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt'
ann_file_val       = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt'
ann_file_unlabeled = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/unlabeled.txt'

# ——— 2. 原始帧根目录 ———
data_prefix_rgb  = {'img': r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/rawframes'}
data_prefix_ndvi = {'img': r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/all_ndvi'}

# ——— 3. 半监督超参 ———
lambda_u  = 1.0
ema_decay = 0.99

# ——— 4. 从 train_loader 拿 clip_len 和 batch_size ———
train_dataloader = {
    'batch_size': 4,
    'dataset': {
        'pipeline': [
            {'type': 'SampleFrames', 'clip_len': 20},  # 跟你脚本里 SampleFrames.clip_len 一致
        ]
    }
}

# ——— 5. 优化 & 轮数 & 类别数 ———
optim_wrapper = {'optimizer': {'lr': 0.001}}
train_cfg     = {'max_epochs': 50}
model         = {'cls_head': {'num_classes': 2}}

# ——— 6. 输出保存目录 ———
work_dir = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/teacher_student'