TRAIN_REID = True
_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]
model = dict(
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'  # noqa: E501
        )))
# dataset settings
# data_root = 'data/MOT17_tiny/'
# data_root = 'mmtracking/data/MOT17_tiny/'
data_root = 'mmtracking/data/MOT17/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/train_80.txt'),
        # ann_file=data_root + 'reid/meta/train_9.txt'),
    val=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'),
    test=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'))

# optimizer
optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[5])
# runtime settings
device = "cuda"
total_epochs = 6
