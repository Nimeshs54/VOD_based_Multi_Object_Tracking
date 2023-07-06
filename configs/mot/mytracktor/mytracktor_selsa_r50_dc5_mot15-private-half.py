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
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth'
        # )
        ),
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
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
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

# # dataset settings
# data = dict(
#     val=dict(
#         ref_img_sampler=dict(
#             _delete_=True,
#             num_ref_imgs=14,
#             frame_range=[-7, 7],
#             method='test_with_adaptive_stride')),
#     test=dict(
#         ref_img_sampler=dict(
#             _delete_=True,
#             num_ref_imgs=14,
#             frame_range=[-7, 7],
#             method='test_with_adaptive_stride')))

# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=100,
#     warmup_ratio=1.0 / 100,
#     step=[3])
# # runtime settings
# total_epochs = 4
# evaluation = dict(metric=['bbox', 'track'], interval=1)
# search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# data
data_root = 'data/MOT17_tiny/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']


# data_root = 'data/MOT17_tiny/'
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='MOTChallengeDataset',
#         visibility_thr=-1,
#         ann_file='data/MOT17_tiny/annotations/half-train_cocoformat.json',
#         img_prefix='data/MOT17_tiny/train',
#         ref_img_sampler=dict(
#             num_ref_imgs=1,
#             frame_range=10,
#             filter_key_img=True,
#             method='uniform'),
#         pipeline=[
#             dict(type='LoadMultiImagesFromFile', to_float32=True),
#             dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
#             dict(
#                 type='SeqResize',
#                 img_scale=(1088, 1088),
#                 share_params=True,
#                 ratio_range=(0.8, 1.2),
#                 keep_ratio=True,
#                 bbox_clip_border=False),
#             dict(type='SeqPhotoMetricDistortion', share_params=True),
#             dict(
#                 type='SeqRandomCrop',
#                 share_params=False,
#                 crop_size=(1088, 1088),
#                 bbox_clip_border=False),
#             dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
#             dict(
#                 type='SeqNormalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='SeqPad', size_divisor=32),
#             dict(type='MatchInstances', skip_nomatch=True),
#             dict(
#                 type='VideoCollect',
#                 keys=[
#                     'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
#                     'gt_instance_ids'
#                 ]),
#             dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
#         ]),
#     val=dict(
#         type='MOTChallengeDataset',
#         ann_file='data/MOT17_tiny/annotations/half-val_cocoformat.json',
#         img_prefix='data/MOT17_tiny/train',
#         ref_img_sampler=dict(
#             num_ref_imgs=14,
#             frame_range=[-7, 7],
#             method='test_with_adaptive_stride'),
#         pipeline=[
#             dict(type='LoadMultiImagesFromFile'),
#             dict(type='SeqResize', img_scale=(1088, 1088), keep_ratio=True),
#             dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
#             dict(
#                 type='SeqNormalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='SeqPad', size_divisor=32),
#             dict(
#                 type='VideoCollect',
#                 keys=['img'],
#                 meta_keys=('num_left_ref_imgs', 'frame_stride')),
#             dict(type='ConcatVideoReferences'),
#             dict(type='MultiImagesToTensor', ref_prefix='ref'),
#             dict(type='ToList')
#         ]),
#     test=dict(
#         type='MOTChallengeDataset',
#         ann_file='data/MOT17_tiny/annotations/half-val_cocoformat.json',
#         img_prefix='data/MOT17_tiny/train',
#         ref_img_sampler=dict(
#             num_ref_imgs=14,
#             frame_range=[-7, 7],
#             method='test_with_adaptive_stride'),
#         pipeline=[
#             dict(type='LoadMultiImagesFromFile'),
#             dict(type='SeqResize', img_scale=(1088, 1088), keep_ratio=True),
#             dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
#             dict(
#                 type='SeqNormalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='SeqPad', size_divisor=32),
#             dict(
#                 type='VideoCollect',
#                 keys=['img'],
#                 meta_keys=('num_left_ref_imgs', 'frame_stride')),
#             dict(type='ConcatVideoReferences'),
#             dict(type='MultiImagesToTensor', ref_prefix='ref'),
#             dict(type='ToList')
#         ]))
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# checkpoint_config = dict(interval=1)
# log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
# opencv_num_threads = 0
# mp_start_method = 'fork'
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=100,
#     warmup_ratio=0.01,
#     step=[3])
# total_epochs = 4
# evaluation = dict(metric=['bbox', 'track'], interval=1)
# search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
