# run mot demo
import mmcv
import tempfile
from mmtrack.apis import inference_mot, init_model, inference_mot_with_vid

# mot_config = './configs/mot/tracktor/tracktor_selsa_r50_fpn_4e_mot15-private-half.py'
# mot_config = './configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py'
mot_config = 'mmtracking/configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py'

# mot_config = 'mmtracking/configs/mot/mytracktor/mytracktor_selsa_r50_dc5_mot15-private-half.py'
# mot_config = 'configs/mot/mytracktor/mytracktor_selsa_r50_dc5_mot15-private-half.py'

# input_video = './demo/demo.mp4'
input_video = 'mmtracking/demo/demo.mp4'
imgs = mmcv.VideoReader(input_video)
mot_model = init_model(mot_config, device='cuda:0')
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
# test and show/save the images
for i, img in enumerate(imgs):
    result = inference_mot(mot_model, img, frame_id=i)
    mot_model.show_result(
        img,
        result,
        show=False,
        wait_time=int(1000. / imgs.fps),
        out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()

# output = './demo/mot1232.mp4'
output = 'mmtracking/demo/mot1zzz.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()


# python demo/demo_mot_vis.py ./configs/mot/tracktor/tracktor_selsa_r50_fpn_4e_mot15-private-half.py --input demo/demo.mp4 --output mot.mp4
# python mmtracking/demo/demo_mot_vis.py ./mmtracking/configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py --input mmtracking/demo/demo.mp4 --output mmtracking/demo/mot369.mp4


# python demo/demo_vid.py ./configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py --input demo/demo.mp4 --checkpoint ./checkpoints/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth --output ./demo/vid.mp4

# python tools/train.py configs/mot/mytracktor/mytracktor_selsa_r50_dc5_mot15-private-half.py

# python tools/train.py configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py

# python tools/analysis/print_config.py configs/mot/mytracktor/mytracktor_selsa_r50_dc5_mot15-private-half.py

# python tools/analysis/print_config.py configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py



# python tools/analysis/print_config.py configs/vid/selsa/selsa_faster_rcnn_mot17.py
# python tools/analysis/print_config.py configs/reid/resnet50_track_MOT17.py


# python tools/train.py configs/mot/mytracktor/mytracktor_selsa_r50_dc5_mot15-private-half.py

# python tools/train.py ./configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py

# python tools/train.py ./configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py

# bash ./tools/dist_train.sh ./configs/vid/selsa/selsa_faster_rcnn_mot17.py 8 \ --work-dir ./work_dirs/

# python tools/train.py configs/vid/selsa/selsa_faster_rcnn_mot17.py
# python tools/train.py configs/reid/resnet50_track_MOT17.py

# bash ./tools/dist_train.sh ./configs/reid/resnet50_track_MOT17.py 8 --work-dir ./work_dirs/
# bash ./tools/dist_train.sh ./configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py 8 --work-dir ./work_dirs/