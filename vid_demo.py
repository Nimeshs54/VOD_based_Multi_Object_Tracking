import mmcv
import tempfile
from mmtrack.apis import inference_vid, init_model

# vid_config = './configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py'
vid_config = 'mmtracking/configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py'
# vid_checkpoint = './checkpoints/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth'
vid_checkpoint = 'mmtracking/checkpoints/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth'

# build the model from a config file and a checkpoint file
vid_model = init_model(vid_config, vid_checkpoint, device='cuda:0')
# input_video = './demo/demo.mp4'
input_video = 'mmtracking/demo/demo.mp4'
imgs = mmcv.VideoReader(input_video)
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
for i, img in enumerate(imgs):
    result = inference_vid(vid_model, img, frame_id=i)
    vid_model.show_result(
            img,
            result,
            wait_time=int(1000. / imgs.fps),
            out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()
# output = './demo/vid.mp4'
output = 'mmtracking/demo/vid111.mp4'
print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()