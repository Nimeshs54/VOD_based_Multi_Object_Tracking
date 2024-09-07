_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/mot_challenge.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='MyTracktor',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_classes=1,
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='./work_dirs/vid_detector/epoch_7.pth'
            checkpoint='./mmtracking/work_dirs/vid_detector/epoch_7-cpy.pth'
            # checkpoint='./mmtracking/work_dirs/vid_det_mot17/epoch_7-cpy.pth'
            # checkpoint=# noqa: E251
            # './mmtracking/work_dirs/selsa_det_mot20/epoch_5.pth'  # noqa: E501
            # '/home/nimesh/mmtracking/work_dirs/selsa_det_mot20/epoch_5-cpy.pth'  # noqa: E501
        )),
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
            # checkpoint='./work_dirs/reid/epoch_3.pth'
            checkpoint='./mmtracking/work_dirs/reid/epoch_2.pth'
            # checkpoint='./mmtracking/work_dirs/reid_mot17/epoch_2.pth'
            # checkpoint=# noqa: E251
            # './mmtracking/work_dirs/reid_mot20/reid_r50_6e_mot20.pth'  # noqa: E501
        )),
    motion=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=0.5,
        regression=dict(
            obj_score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.6),
            match_iou_thr=0.3),
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            match_iou_thr=0.2),
        momentums=None,
        num_frames_retain=10))

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    # dict(type='ConcatVideoReferences'),
    dict(type='ConcatSameTypeFrames'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]

# data
data_root = 'mmtracking/data/MOT17/'

data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        # ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        # ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        # ann_file=data_root + 'annotations/test_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline,
        test_mode=True))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
# load_from='./mmtracking/work_dirs/selsa_det_mot20/epoch_5.pth'
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
