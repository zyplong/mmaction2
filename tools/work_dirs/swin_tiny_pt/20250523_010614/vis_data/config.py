ann_train = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt'
ann_val = 'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt'
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.seasonal_pseudo_change.downstream.models.load_pt',
        'projects.seasonal_pseudo_change.downstream.pipelines.load_pt_feature',
    ])
data_root = 'projects/seasonal_pseudo_change/datasets/pt_spatial_features'
dataset_type = 'PTFeatureDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.1,
        in_channels=2048,
        patch_norm=True,
        patch_size=(
            2,
            4,
            4,
        ),
        qkv_bias=True,
        type='SwinTransformer3D',
        window_size=(
            8,
            7,
            7,
        )),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=768,
        num_classes=2,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            0,
            0,
            0,
        ],
        std=[
            1,
            1,
            1,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.02),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=2.5, start_factor=0.1, type='LinearLR'),
    dict(T_max=30, begin=0, by_epoch=True, end=30, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt',
        data_prefix=
        'projects/seasonal_pseudo_change/datasets/pt_spatial_features',
        pipeline=[
            dict(type='LoadPTFeature'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='PTFeatureDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(type='LoadPTFeature'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/train.txt',
        data_prefix=
        'projects/seasonal_pseudo_change/datasets/pt_spatial_features',
        pipeline=[
            dict(type='LoadPTFeature'),
            dict(type='PackActionInputs'),
        ],
        type='PTFeatureDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadPTFeature'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'F:/zyp/Thesis source code/mmaction2/projects/seasonal_pseudo_change/datasets/splits/val.txt',
        data_prefix=
        'projects/seasonal_pseudo_change/datasets/pt_spatial_features',
        pipeline=[
            dict(type='LoadPTFeature'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='PTFeatureDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(type='LoadPTFeature'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/swin_tiny_pt'
