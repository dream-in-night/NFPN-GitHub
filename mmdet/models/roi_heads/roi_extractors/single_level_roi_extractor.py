import torch

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor
import torch.nn as nn
import torch
eps=0.0001
@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 init=0.5,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
        self.finest_scale = finest_scale
    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls
    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, feats1, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        # print("num_levels:",num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        # print("in SingleRoIExtractor:")
        # print("roi_feats size:",roi_feats.size())#[512, 256, 7, 7]
        # print("rois size:",rois.size())#[512, 5]
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t1 = self.roi_layers1[i](feats1[i], rois_)
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t + roi_feats_t1
                # roi_feats[inds] = roi_feats_t + roi_feats_t1
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
        return roi_feats
    # @force_fp32(apply_to=('feats', ), out_fp16=True)
    # def forward(self, feats, feats1, feats2, rois, roi_scale_factor=None):
    #     """Forward function."""
    #     out_size = self.roi_layers[0].output_size
    #     num_levels = len(feats)
    #     # print("num_levels:",num_levels)
    #     roi_feats = feats[0].new_zeros(
    #         rois.size(0), self.out_channels, *out_size)
    #     # TODO: remove this when parrots supports
    #     # print("in SingleRoIExtractor:")
    #     # print("roi_feats size:",roi_feats.size())#[512, 256, 7, 7]
    #     # print("rois size:",rois.size())#[512, 5]
    #     if torch.__version__ == 'parrots':
    #         roi_feats.requires_grad = True

    #     if num_levels == 1:
    #         if len(rois) == 0:
    #             return roi_feats
    #         return self.roi_layers[0](feats[0], rois)

    #     target_lvls = self.map_roi_levels(rois, num_levels)
    #     if roi_scale_factor is not None:
    #         rois = self.roi_rescale(rois, roi_scale_factor)
    #     for i in range(num_levels):
    #         inds = target_lvls == i
    #         if inds.any():
    #             rois_ = rois[inds, :]
    #             if i==3:
    #                 roi_feats_t2 = self.roi_layers2[i](feats2[i], rois_)
    #                 roi_feats[inds] = roi_feats_t2
    #             if i==2:
    #                 roi_feats_t2 = self.roi_layers2[i](feats2[i], rois_)
    #                 roi_feats_t1 = self.roi_layers1[i](feats1[i], rois_)
    #                 roi_feats[inds] = roi_feats_t2 + roi_feats_t1
    #             if i==1:
    #                 roi_feats_t = self.roi_layers[i](feats[i], rois_)
    #                 roi_feats_t1 = self.roi_layers1[i](feats1[i], rois_)
    #                 roi_feats[inds] = roi_feats_t + roi_feats_t1
    #             if i==0:
    #                 roi_feats_t = self.roi_layers[i](feats[i], rois_)
    #                 roi_feats[inds] = roi_feats_t#roi2_bfp4_weightroi
    #             # roi_feats[inds] = roi_feats_t + roi_feats_t1
    #         else:
    #             roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
    #     return roi_feats
    # @force_fp32(apply_to=('feats', ), out_fp16=True)
    # def forward(self, feats, rois, roi_scale_factor=None):
    #     """Forward function."""
    #     out_size = self.roi_layers[0].output_size
    #     num_levels = len(feats)
    #     roi_feats = feats[0].new_zeros(
    #         rois.size(0), self.out_channels, *out_size)
    #     # TODO: remove this when parrots supports
    #     # print("in SingleRoIExtractor:")
    #     # print("roi_feats size:",roi_feats.size())#[512, 256, 7, 7]
    #     # print("rois size:",rois.size())#[512, 5]
    #     if torch.__version__ == 'parrots':
    #         roi_feats.requires_grad = True

    #     if num_levels == 1:
    #         if len(rois) == 0:
    #             return roi_feats
    #         return self.roi_layers[0](feats[0], rois)

    #     target_lvls = self.map_roi_levels(rois, num_levels)
    #     if roi_scale_factor is not None:
    #         rois = self.roi_rescale(rois, roi_scale_factor)
    #     for i in range(num_levels):
    #         inds = target_lvls == i
    #         if inds.any():
    #             rois_ = rois[inds, :]
    #             roi_feats_t = self.roi_layers[i](feats[i], rois_)
    #             roi_feats[inds] = roi_feats_t
    #         else:
    #             roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
    #     return roi_feats
