ann_file_test = 'projects/seasonal_pseudo_change/datasets/splits/val.csv'
ann_file_train = 'projects/seasonal_pseudo_change/datasets/splits/train.csv'
ann_file_val = 'projects/seasonal_pseudo_change/datasets/splits/val.csv'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'projects/seasonal_pseudo_change/datasets/rawframes'
data_root_val = 'projects/seasonal_pseudo_change/datasets/rawframes'
dataset_type = 'RawframeDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=10, type='LoggerHook'),
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
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        arch='tiny',
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        mlp_ratio=4.0,
        patch_norm=True,
        patch_size=(
            2,
            4,
            4,
        ),
        pretrained=
        'projects/seasonal_pseudo_change/pretrained/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth',
        pretrained2d=True,
        qk_scale=None,
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
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    constructor='SwinOptimWrapperConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.02),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2.5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=30,
        begin=0,
        by_epoch=True,
        end=30,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='projects/seasonal_pseudo_change/datasets/splits/val.csv',
        data_prefix=dict(
            img='projects/seasonal_pseudo_change/datasets/rawframes'),
        pipeline=[
            dict(type='RawFrameDecode'),
            dict(
                clip_len=8,
                frame_interval=1,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(type='RawFrameDecode'),
    dict(
        clip_len=8,
        frame_interval=1,
        num_clips=4,
        test_mode=True,
        type='SampleFrames'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=1, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='projects/seasonal_pseudo_change/datasets/splits/train.csv',
        data_prefix=dict(
            img='projects/seasonal_pseudo_change/datasets/rawframes'),
        pipeline=[
            dict(type='RawFrameDecode'),
            dict(
                clip_len=8, frame_interval=1, num_clips=1,
                type='SampleFrames'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='RawframeDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='RawFrameDecode'),
    dict(clip_len=8, frame_interval=1, num_clips=1, type='SampleFrames'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='projects/seasonal_pseudo_change/datasets/splits/val.csv',
        data_prefix=dict(
            img='projects/seasonal_pseudo_change/datasets/rawframes'),
        pipeline=[
            dict(type='RawFrameDecode'),
            dict(
                clip_len=8,
                frame_interval=1,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(type='RawFrameDecode'),
    dict(
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\swin_tiny_2class'
