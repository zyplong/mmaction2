_base_ = [
    '../_base_/models/swin_tiny.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained='projects/seasonal_pseudo_change/pretrained/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob')
)

# dataset settings
# ✅ 修改为你本地的数据集路径（rawframes + csv 文件）
dataset_type = 'RawframeDataset'
data_root = 'projects/seasonal_pseudo_change/datasets/rawframes'
data_root_val = data_root
ann_file_train = 'projects/seasonal_pseudo_change/datasets/splits/train.csv'
ann_file_val = 'projects/seasonal_pseudo_change/datasets/splits/val.csv'
ann_file_test = ann_file_val

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='RawFrameDecode'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='RawFrameDecode'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1, test_mode=True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='RawFrameDecode'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=4, test_mode=True),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,  # ✅ 先快速测试1轮
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1),
    logger=dict(interval=10))

auto_scale_lr = dict(enable=False, base_batch_size=64)
