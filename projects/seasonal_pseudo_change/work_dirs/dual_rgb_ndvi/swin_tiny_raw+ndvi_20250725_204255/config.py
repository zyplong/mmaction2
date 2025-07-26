# configs/seasonal_pseudo_change/dualbranch_paths_only.py
_base_ = ['../../configs/_base_/default_runtime.py']

# —— 有标签训练/验证/无标签文件
ann_file_train     = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt'
ann_file_val       = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt'
ann_file_unlabeled = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/unlabeled.txt'

# —— 原始帧目录各自独立 ——
#    （rawframes 里只有 RGB .jpg，all_ndvi 里只有 NDVI .tif）
data_prefix_rgb  = dict(img='F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/rawframes')
data_prefix_ndvi = dict(img='F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/all_ndvi')

# —— 保留旧 logits（不影响此例）
ddpm_logits_dir = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/results/logits'

# -------------------------------
# 2️⃣ 模型：双支路 Late-Fusion
# -------------------------------
model = dict(
    type='Recognizer3D',
    # — RGB 支路 —
    rgb_backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        in_channels=3,
        pretrained=(
            'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/'
            'pretrained/videoswin/swin_tiny_patch244_window877_kinetics400_1k_state_dict.pth'
        ),
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True
    ),
    # — NDVI 支路 —
    ndvi_backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        in_channels=1,
        pretrained=None,  # NDVI 一般用随机初始化
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True
    ),
    # Late-Fusion：concat 或 add
    fuse_type='concat',
    # concat 后的分类头
    cls_head=dict(
        type='I3DHead',
        in_channels=768 * 2,  # 768(rgb) + 768(ndvi)
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        format_shape='NCTHW',
        # 只对 RGB 做 normalize，NDVI 在 pipeline 里单独归一化
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    )
)

# -------------------------------
# 3️⃣ DataLoader
# -------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# —— 训练集（有标签）——
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DualBranchDataset',
        rgb_dataset = dict(
            type='RawframeDataset',
            ann_file=ann_file_train,
            data_prefix=data_prefix_rgb,
            filename_tmpl='img_{:04d}.jpg',
            pipeline=[
                dict(type='SampleFrames', clip_len=20, frame_interval=1, num_clips=1),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(-1,256)),
                dict(type='RandomResizedCrop'),
                dict(type='Resize', scale=(224,224), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='PackActionInputs'),
            ]
        ),
        ndvi_dataset = dict(
            type='RawframeDataset',
            ann_file=ann_file_train,
            data_prefix=data_prefix_ndvi,
            filename_tmpl='img_{:04d}.tif',
            pipeline=[
                dict(type='SampleFrames', clip_len=20, frame_interval=1, num_clips=1),
                dict(type='RawFrameDecode', color_type='unchanged'),  # 读１通道
                dict(type='Resize', scale=(-1,256)),
                dict(type='RandomResizedCrop'),
                dict(type='Resize', scale=(224,224), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='Normalize', mean=[127.0], std=[50.0]),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='PackActionInputs'),
            ]
        )
    )
)

# —— 验证 & 测试 ——（同理）
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DualBranchDataset',
        rgb_dataset = dict(
            type='RawframeDataset',
            ann_file=ann_file_val,
            data_prefix=data_prefix_rgb,
            filename_tmpl='img_{:04d}.jpg',
            pipeline=[
                dict(type='SampleFrames', clip_len=20, frame_interval=1, num_clips=1, test_mode=True),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(-1,256)),
                dict(type='CenterCrop', crop_size=224),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='PackActionInputs'),
            ]
        ),
        ndvi_dataset = dict(
            type='RawframeDataset',
            ann_file=ann_file_val,
            data_prefix=data_prefix_ndvi,
            filename_tmpl='img_{:04d}.tif',
            pipeline=[
                dict(type='SampleFrames', clip_len=20, frame_interval=1, num_clips=1, test_mode=True),
                dict(type='RawFrameDecode', color_type='unchanged'),
                dict(type='Resize', scale=(-1,256)),
                dict(type='CenterCrop', crop_size=224),
                dict(type='Normalize', mean=[127.0], std=[50.0]),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='PackActionInputs'),
            ]
        )
    )
)
test_dataloader = val_dataloader

# —— 无标签半监督 —— 重用上面 pipeline
unlabeled_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DualBranchDataset',
        rgb_dataset = train_dataloader['dataset']['rgb_dataset'],
        ndvi_dataset = train_dataloader['dataset']['ndvi_dataset'],
        # 只把 ann_file 换成 unlabeled
        rgb_dataset_copy = dict(ann_file=ann_file_unlabeled),
        ndvi_dataset_copy= dict(ann_file=ann_file_unlabeled),
    )
)

# -------------------------------
# 4️⃣ 优化器 & 调度
# -------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.9,0.999), weight_decay=0.02),
    paramwise_cfg=dict(
        norm=dict(decay_mult=0.0),
        bias=dict(decay_mult=0.0),
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
    ),
    loss_scale='dynamic'
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=48, eta_min=0, by_epoch=True, begin=2, end=50),
]

# -------------------------------
# 5️⃣ 日志 & 保存
# -------------------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10),
)
visualizer = dict(type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
work_dir = r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/work_dirs/dual_rgb_ndvi'