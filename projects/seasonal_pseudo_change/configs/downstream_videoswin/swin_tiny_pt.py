# ===== swin_tiny_pt.py =====

# 1. 在启动时自动 import 你自定义的 Dataset & Transform
# configs/downstream_videoswin/swin_tiny_pt.py
# custom_imports = dict(
#     imports=[
#         'projects.seasonal_pseudo_change.downstream.models.load_pt',
#         'projects.seasonal_pseudo_change.downstream.pipelines.load_pt_feature',
#     ],
#     allow_failed_imports=False,
# )
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.seasonal_pseudo_change.downstream.models.load_pt',
        'projects.seasonal_pseudo_change.downstream.pipelines.load_pt_feature',
        'projects.seasonal_pseudo_change.downstream.pipelines.pack_pt_inputs',  # ✅ 新增
    ])

_base_ = []

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        in_channels=2048,
        drop_path_rate=0.1,
        patch_norm=True,
        qkv_bias=True,

),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob',
    ),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=None,
        std=None,
        type='ActionDataPreprocessor')

)


dataset_type = 'PTFeatureDataset'
# train_pipeline = [
#     dict(type='LoadPTFeature'),
#     dict(type='PackActionInputs'),
# ]
train_pipeline = [
    dict(type='LoadPTFeature'),
    dict(type='PackPTInputs'),
]

val_pipeline = train_pipeline
test_pipeline = train_pipeline

# 下面三个都要改成字符串
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt',
        data_prefix=r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/pt_spatial_features',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt',
        data_prefix=r'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/pt_spatial_features',
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# ...（下面不变）


# 4. 优化器 & 学习率调度
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        norm=dict(decay_mult=0.0),
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2.5),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=30, T_max=30),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# 5. 其他通用设置
default_scope = 'mmaction'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
)

# ...
log_level = 'INFO'
resume = False


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

work_dir = './work_dirs/swin_tiny_pt'