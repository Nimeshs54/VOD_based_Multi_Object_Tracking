# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import cv2
import mmcv
import os

from mmdet.models import build_detector

from mmtrack.core import outs2results
import numpy as np
from ..builder import MODELS, build_motion, build_reid, build_tracker
from ..motion import CameraMotionCompensation, LinearMotion
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class Tracktor(BaseMultiObjectTracker):
    """Tracking without bells and whistles.

    Details can be found at `Tracktor<https://arxiv.org/abs/1903.05625>`_.
    """

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if detector:
                detector_pretrain = pretrains.get('detector', None)
                if detector_pretrain:
                    detector.init_cfg = dict(
                        type='Pretrained', checkpoint=detector_pretrain)
                else:
                    detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None
        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if motion is not None:
            self.motion = build_motion(motion)
            if not isinstance(self.motion, list):
                self.motion = [self.motion]
            for m in self.motion:
                if isinstance(m, CameraMotionCompensation):
                    self.cmc = m
                if isinstance(m, LinearMotion):
                    self.linear_motion = m

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    @property
    def with_cmc(self):
        """bool: whether the framework has a camera model compensation
                model.
        """
        return hasattr(self, 'cmc') and self.cmc is not None

    @property
    def with_linear_motion(self):
        """bool: whether the framework has a linear motion model."""
        return hasattr(self,
                       'linear_motion') and self.linear_motion is not None

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with Tracktor.')

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)

        # for a in x:
        #     print(a.shape)

        # Assume the tuple is named 'tensor_tuple'
        # for idx, x_tensor in enumerate(x):
        #     print(f"Processing tensor {idx}...")
        #     print(f"Input tensor shape: {x_tensor.shape}")

        #     # Convert the tensor to a numpy array
        #     tensor_np = x_tensor.cpu().numpy()
        #     print(f"Output numpy array shape: {tensor_np.shape}")

        #     # Reshape the array to a 2D image
        #     tensor_img = tensor_np.reshape(
        #         tensor_np.shape[1], tensor_np.shape[2], tensor_np.shape[3], 1)
        #     print(f"Output numpy array shape: {tensor_img.shape}")

        #     # Scale the pixel values to the range [0, 255]
        #     tensor_img = (tensor_img - tensor_img.min()) * \
        #         (255 / (tensor_img.max() - tensor_img.min()))

        #     # convert the numpy array to the uint8 data type
        #     tensor_img = tensor_img.astype('uint8')

        #     # # resize the image to a smaller size
        #     # resized_arr = cv2.resize(tensor_img, (512, 512), interpolation=cv2.INTER_NEAREST)

        #     # Save the image using OpenCV
        #     output_path = 'mmtracking/demo/'
        #     filename = os.path.join(output_path, f'tensor_{idx}.png')
        #     cv2.imwrite(filename, tensor_img)




        if hasattr(self.detector, 'roi_head'):
            # TODO: check whether this is the case
            if public_bboxes is not None:
                public_bboxes = [_[0] for _ in public_bboxes]
                proposals = public_bboxes
            else:
                proposals = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)
                # print(proposals[0].shape)

                # proposals_np =  proposals[0].cpu().numpy()
                # proposals_np = (proposals_np - np.min(proposals_np)) * 255 / (np.max(proposals_np) - np.min(proposals_np))
                # proposals_np = proposals_np.astype(np.uint8)
                # image = np.reshape(proposals_np, (1000, 5, 1))
                # cv2.imwrite("mmtracking/demo/tensor_image.jpg", image)

                # x_np =  x[0].cpu().numpy()
                # x_np = np.transpose(x_np, (1, 2, 3, 0))
                # x_np = (x_np - np.min(x_np)) * 255 / (np.max(x_np) - np.min(x_np))
                # x_np = x_np.astype(np.uint8)
                # cv2.imwrite("mmtracking/demo/x_image.jpg", x_np)

                # print(type(x))
                # print(type(proposals))

            

            det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposals,
                self.detector.roi_head.test_cfg,
                rescale=rescale)
            # TODO: support batch inference
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
            num_classes = self.detector.roi_head.bbox_head.num_classes
        elif hasattr(self.detector, 'bbox_head'):
            num_classes = self.detector.bbox_head.num_classes
            raise NotImplementedError(
                'Tracktor must need "roi_head" to refine proposals.')
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
