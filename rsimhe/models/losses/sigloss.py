# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from rsimhe.models.builder import LOSSES

@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_rsimhe=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_rsimhe = max_rsimhe

        self.eps = 0.1 # avoid grad explode

        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_rsimhe is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_rsimhe)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self,
                rsimhe_pred,
                rsimhe_gt,
                **kwargs):
        """Forward function."""
        
        loss_rsimhe = self.loss_weight * self.sigloss(
            rsimhe_pred,
            rsimhe_gt,
            )
        return loss_rsimhe
