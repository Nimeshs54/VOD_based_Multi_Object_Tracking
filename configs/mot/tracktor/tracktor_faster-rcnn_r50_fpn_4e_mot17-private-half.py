# _base_ = [
#     '../../_base_/models/faster_rcnn_r50_fpn.py',
#     '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
# ]

# model = dict(
#     type='Tracktor',
#     detector=dict(
#         rpn_head=dict(bbox_coder=dict(clip_border=False)),
#         roi_head=dict(
#             bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
#         init_cfg=dict(
#             type='Pretrained',
#             # checkpoint=  # noqa: E251
#             # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
#             checkpoint=  # noqa: E251
#             './mmtracking/work_dirs/selsa_det_mot20/epoch_5.pth'  # noqa: E501
#             # 'mmtracking/work_dirs/det/epoch_4.pth'  # noqa: E501
#         )),
#     reid=dict(
#         type='BaseReID',
#         backbone=dict(
#             type='ResNet',
#             depth=50,
#             num_stages=4,
#             out_indices=(3, ),
#             style='pytorch'),
#         neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
#         head=dict(
#             type='LinearReIDHead',
#             num_fcs=1,
#             in_channels=2048,
#             fc_channels=1024,
#             out_channels=128,
#             num_classes=380,
#             loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#             loss_pairwise=dict(
#                 type='TripletLoss', margin=0.3, loss_weight=1.0),
#             norm_cfg=dict(type='BN1d'),
#             act_cfg=dict(type='ReLU')),
#         init_cfg=dict(
#             type='Pretrained',
#             # checkpoint=  # noqa: E251
#             # 'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
#             checkpoint=  # noqa: E251
#             './mmtracking/work_dirs/reid_mot20/reid_r50_6e_mot20.pth'  # noqa: E501
#             # './mmtracking/work_dirs/reid/epoch_2.pth'  # noqa: E501/
#         )),
#     motion=dict(
#         type='CameraMotionCompensation',
#         warp_mode='cv2.MOTION_EUCLIDEAN',
#         num_iters=100,
#         stop_eps=0.00001),
#     tracker=dict(
#         type='TracktorTracker',
#         obj_score_thr=0.5,
#         regression=dict(
#             obj_score_thr=0.5,
#             nms=dict(type='nms', iou_threshold=0.6),
#             match_iou_thr=0.3),
#         reid=dict(
#             num_samples=10,
#             img_scale=(256, 128),
#             img_norm_cfg=None,
#             match_score_thr=2.0,
#             match_iou_thr=0.2),
#         momentums=None,
#         num_frames_retain=10))

# # data_root = 'mmtracking/data/MOT17_tiny/'
# data_root = 'mmtracking/data/MOT20/'
# data = dict(
#     train=dict(
#         ann_file=data_root + 'annotations/half-train_cocoformat.json',
#         # ann_file=data_root + 'annotations/train_cocoformat.json',
#         img_prefix=data_root + 'train'),
#     val=dict(
#         ann_file=data_root + 'annotations/half-val_cocoformat.json',
#         # ann_file=data_root + 'annotations/train_cocoformat.json',
#         img_prefix=data_root + 'train'),
#     test=dict(
#         # ann_file=data_root + 'annotations/test_cocoformat.json',
#         # img_prefix=data_root + 'train'))
#         ann_file=data_root + 'annotations/half-val_cocoformat.json',
#         img_prefix=data_root + 'train'))

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


_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='Tracktor',
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=# noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
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
            checkpoint=# noqa: E251
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

data_root = 'mmtracking/data/MOT17/'
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
