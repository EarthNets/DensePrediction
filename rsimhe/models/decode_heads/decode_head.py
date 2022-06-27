
import mmcv
import copy
import torch

import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from rsimhe.ops import resize
from rsimhe.models.builder import build_loss


class DepthBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (List): Input channels.
        channels (int): Channels after modules, before conv_rsimhe.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        loss_decode (dict): Config of decode loss.
            Default: dict(type='SigLoss').
        sampler (dict|None): The config of rsimhe map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        min_rsimhe (int): Min rsimhe in dataset setting.
            Default: 1e-3.
        max_rsimhe (int): Max rsimhe in dataset setting.
            Default: None.
        norm_cfg (dict|None): Config of norm layers.
            Default: None.
        classify (bool): Whether predict rsimhe in a cls.-reg. manner.
            Default: False.
        n_bins (int): The number of bins used in cls. step.
            Default: 256.
        bins_strategy (str): The discrete strategy used in cls. step.
            Default: 'UD'.
        norm_strategy (str): The norm strategy on cls. probability 
            distribution. Default: 'linear'
        scale_up (str): Whether predict rsimhe in a scale-up manner.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels=96,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='SigLoss',
                     valid_mask=True,
                     loss_weight=10),
                 sampler=None,
                 align_corners=False,
                 min_rsimhe=1e-3,
                 max_rsimhe=None,
                 norm_cfg=None,
                 classify=False,
                 n_bins=256,
                 bins_strategy='UD',
                 norm_strategy='linear',
                 scale_up=False,
                 ):
        super(DepthBaseDecodeHead, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.align_corners = align_corners
        self.min_rsimhe = min_rsimhe
        self.max_rsimhe = max_rsimhe
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up

        if self.classify:
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.softmax = nn.Softmax(dim=1)
            self.conv_rsimhe = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_rsimhe = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)
        

        self.fp16_enabled = False
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def extra_repr(self):
        """Extra repr."""
        s = f'align_corners={self.align_corners}'
        return s

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass
    
    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, rsimhe_gt, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsimhe/datasets/pipelines/formatting.py:Collect`.
            rsimhe_gt (Tensor): GT rsimhe
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        rsimhe_pred = self.forward(inputs, img_metas)
        losses = self.losses(rsimhe_pred, rsimhe_gt)

        log_imgs = self.log_images(img[0], rsimhe_pred[0], rsimhe_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsimhe/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output rsimhe map.
        """
        return self.forward(inputs, img_metas)

    def rsimhe_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_rsimhe(feat)

            if self.bins_strategy == 'UD':
                bins = torch.linspace(self.min_rsimhe, self.max_rsimhe, self.n_bins, device=feat.device)
            elif self.bins_strategy == 'SID':
                bins = torch.logspace(self.min_rsimhe, self.max_rsimhe, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == 'linear':
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == 'softmax':
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == 'sigmoid':
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum('ikmn,k->imn', [logit, bins]).unsqueeze(dim=1)

        else:
            if self.scale_up:
                output = self.sigmoid(self.conv_rsimhe(feat)) * self.max_rsimhe
            else:
                output = self.relu(self.conv_rsimhe(feat)) + self.min_rsimhe
        return output

    @force_fp32(apply_to=('rsimhe_pred', ))
    def losses(self, rsimhe_pred, rsimhe_gt):
        """Compute rsimhe loss."""
        loss = dict()
        rsimhe_pred = resize(
            input=rsimhe_pred,
            size=rsimhe_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        loss['loss_rsimhe'] = self.loss_decode(
            rsimhe_pred,
            rsimhe_gt)
        return loss

    def log_images(self, img_path, rsimhe_pred, rsimhe_gt, img_meta):
        show_img = copy.deepcopy(img_path.detach().cpu().permute(1, 2, 0))
        show_img = show_img.numpy().astype(np.float32)
        show_img = mmcv.imdenormalize(show_img, 
                                      img_meta['img_norm_cfg']['mean'],
                                      img_meta['img_norm_cfg']['std'], 
                                      img_meta['img_norm_cfg']['to_rgb'])
        show_img = np.clip(show_img, 0, 255)
        show_img = show_img.astype(np.uint8)
        show_img = show_img[:, :, ::-1]
        show_img = show_img.transpose(0, 2, 1)
        show_img = show_img.transpose(1, 0, 2)

        rsimhe_pred = rsimhe_pred / torch.max(rsimhe_pred)
        rsimhe_gt = rsimhe_gt / torch.max(rsimhe_gt)

        rsimhe_pred_color = copy.deepcopy(rsimhe_pred.detach().cpu())
        rsimhe_gt_color = copy.deepcopy(rsimhe_gt.detach().cpu())

        return {"img_rgb": show_img, "img_rsimhe_pred": rsimhe_pred_color, "img_rsimhe_gt": rsimhe_gt_color}