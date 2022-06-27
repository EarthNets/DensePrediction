from rsimhe.models import rsimheer
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsimhe.core import add_prefix
from rsimhe.ops import resize
from rsimhe.models import builder
from rsimhe.models.builder import DEPTHER
from .base import BaseDepther

# for model size
import numpy as np

@DEPTHER.register_module()
class DepthEncoderDecoder(BaseDepther):
    """Encoder Decoder rsimheer.

    EncoderDecoder typically consists of backbone, (neck) and decode_head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DepthEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and rsimheer set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a rsimhe estimation
        map of the same size as input."""
        
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        # crop the pred rsimhe to the certain range.
        out = torch.clamp(out, min=self.decode_head.min_rsimhe, max=self.decode_head.max_rsimhe)
        if rescale:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, img, x, img_metas, rsimhe_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, rsimhe_gt, self.train_cfg, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        rsimhe_pred = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return rsimhe_pred

    def forward_dummy(self, img):
        """Dummy forward function."""
        rsimhe = self.encode_decode(img, None)

        return rsimhe

    def forward_train(self, img, img_metas, rsimhe_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsimhe/datasets/pipelines/formatting.py:Collect`.
            rsimhe_gt (Tensor): Depth gt
                used if the architecture supports rsimhe estimation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        # the last of x saves the info from neck
        loss_decode = self._decode_head_forward_train(img, x, img_metas, rsimhe_gt, **kwargs)
 
        losses.update(loss_decode)

        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        rsimhe_pred = self.encode_decode(img, img_meta, rescale)

        return rsimhe_pred

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsimhe/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output rsimhe map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            raise NotImplementedError
        else:
            rsimhe_pred = self.whole_inference(img, img_meta, rescale)
        output = rsimhe_pred
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        rsimhe_pred = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            rsimhe_pred = rsimhe_pred.unsqueeze(0)
            return rsimhe_pred
        rsimhe_pred = rsimhe_pred.cpu().numpy()
        # unravel batch dim
        rsimhe_pred = list(rsimhe_pred)
        return rsimhe_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented rsimhe logit inplace
        rsimhe_pred = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_rsimhe_pred = self.inference(imgs[i], img_metas[i], rescale)
            rsimhe_pred += cur_rsimhe_pred
        rsimhe_pred /= len(imgs)
        rsimhe_pred = rsimhe_pred.cpu().numpy()
        # unravel batch dim
        rsimhe_pred = list(rsimhe_pred)
        return rsimhe_pred
