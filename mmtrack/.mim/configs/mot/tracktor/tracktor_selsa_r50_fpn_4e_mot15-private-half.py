_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py']

# _base_ = [
#     '../../_base_/models/selsa_r50.py',
#     '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
# ]

model = dict(
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth')))
