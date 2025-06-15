# 导入基础配置
_base_ = ['../../configs/_base_/default_runtime.py']

# 数据路径配置
ann_file_train = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt'
ann_file_val = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt'
data_root = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/rawframes'
ann_file_unlabeled = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/unlabeled.txt'
data_prefix = dict(img=data_root)

# 模型配置（加载预训练权重）
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/pretrained/videoswin/swin_tiny_patch244_window877_kinetics400_1k_converted.pth',
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True
    ),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        format_shape='NCTHW',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    )
)

# 训练设置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 数据加载器配置
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RawframeDataset',
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        # 在这里指定文件名模板
        filename_tmpl='img_{:04d}.jpg',
        pipeline=[
            dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
            dict(type='RawFrameDecode'),  # 这里不带 filename_tmpl
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RawframeDataset',
        ann_file=ann_file_val,
        data_prefix=data_prefix,
        test_mode=True,
        filename_tmpl='img_{:04d}.jpg',  # 同样在这里指定
        pipeline=[
            dict(type='SampleFrames', clip_len=8, frame_interval=1,
                 num_clips=1, test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]
    )
)

# 如果 test 和 val 一模一样，就直接复用
test_dataloader = val_dataloader

# 半监督无标签数据加载
unlabeled_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RawframeDataset',
        ann_file=ann_file_unlabeled,
        data_prefix=data_prefix,    # 推荐统一成dict(img=data_root)
        filename_tmpl='img_{:04d}.jpg',
        pipeline=[
            dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]
    )
)

# 评估指标
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# 优化器与调度器
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.02),
    paramwise_cfg=dict(
        norm=dict(decay_mult=0.0),
        bias=dict(decay_mult=0.0),
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1)
    ),
    loss_scale='dynamic'
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2.5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=30, eta_min=0, by_epoch=True, begin=0, end=30)
]

# 日志与保存路径
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10),
)

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)

log_level = 'INFO'
load_from = None
resume = False
work_dir = 'F:\zyp\Thesis source code\mmaction2\projects\seasonal_pseudo_change\work_dirs'

