# dataset settings
dataset_type = 'GTADataset'
data_root = '/home/xshadow/GTA_height'
#data_root = '/home/xshadow/xdata/real_city/JAX'
#data_root = '/home/xshadow/xdata/real_city/AHN_Data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsDepth', reduce_zero_label=False),
    dict(type='Resize', img_scale=(480, 640), ratio_range=(1.0, 1.0)),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1920),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=256,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=256,
        img_dir='images/validation_tiny',
        ann_dir='annotations/validation_tiny',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=256,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
