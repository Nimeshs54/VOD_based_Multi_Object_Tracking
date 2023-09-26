_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/mot_challenge.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                num_classes=1,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
    ))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]


# dataset settings
data_root = 'mmtracking/data/MOT17_tiny/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        detection_file=data_root + 'annotations/half-train_detections.pkl',
        # ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('pedestrian', ),
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=4,
            frame_range=15,
            filter_key_img=True,
            method='bilateral_uniform'),
        pipeline=train_pipeline),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        # ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=30,
            frame_range=[-15, 15],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline),
    # detection_file=data_root + 'annotations/train_detections.pkl'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        # ann_file=data_root + 'annotations/test_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=30,
            frame_range=[-15, 15],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# runtime settings
device = "cuda"
total_epochs = 8
evaluation = dict(metric=['bbox'], interval=1)
# test_set = 'test'
