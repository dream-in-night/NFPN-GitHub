import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS


@NECKS.register_module()
class BFP5(nn.Module):
    """BFP2 (Balanced Feature Pyrmamids)

    BFP2 takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP5, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        self.refine3 = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.refine1 = nn.ModuleList()
        for i in range(self.num_levels):
            refine1 = ConvModule(
                self.in_channels,
                self.in_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.refine1.append(refine1)    
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels
        feats = []
        feat = []
        feat.append(inputs[0])
        gather_size = inputs[0].size()[2:]
        gathered2 = F.interpolate(inputs[1], size=gather_size, mode='nearest')
        gathered2 = self.refine1[0](gathered2)
        feat.append(gathered2)
        bsf = sum(feat) / len(feat)
        bsf = self.refine3(bsf)
        feats.append(bsf)
        for i in range(1,len(inputs)-1):
            feat = []
            gather_size = inputs[i].size()[2:]
            gathered1 = F.adaptive_max_pool2d(inputs[i-1], output_size=gather_size)
            feat.append(gathered1)
            gathered2 = F.interpolate(inputs[i+1], size=gather_size, mode='nearest')
            gathered2 = self.refine1[i](gathered2)
            feat.append(gathered2)
            bsf = sum(feat) / len(feat)
            bsf = self.refine3(bsf)
            feats.append(bsf)
        feat = []
        feat.append(inputs[self.num_levels-1])
        gather_size = inputs[self.num_levels-1].size()[2:]
        gathered2 = F.adaptive_max_pool2d(inputs[self.num_levels-2], output_size=gather_size)
        gathered2 = self.refine1[self.num_levels-1](gathered2)
        feat.append(gathered2)
        bsf = sum(feat) / len(feat)
        bsf = self.refine3(bsf)
        feats.append(bsf)        
        # step 3: scatter refined features to multi-levels by a residual path
        feat2s = []
        feat2 = []
        feat2.append(feats[0])
        gather_size = feats[0].size()[2:]
        gathered2 = F.interpolate(feats[1], size=gather_size, mode='nearest')
        gathered2 = self.refine1[0](gathered2)
        feat2.append(gathered2)
        bsf = sum(feat2) / len(feat2)
        bsf = self.refine3(bsf)
        feat2s.append(bsf)
        for i in range(1,len(feats)-1):
            feat2 = []
            gather_size = feats[i].size()[2:]
            gathered1 = F.adaptive_max_pool2d(feats[i-1], output_size=gather_size)
            feat2.append(gathered1)
            gathered2 = F.interpolate(feats[i+1], size=gather_size, mode='nearest')
            gathered2 = self.refine1[i](gathered2)
            feat2.append(gathered2)
            bsf = sum(feat2) / len(feat2)
            bsf = self.refine3(bsf)
            feat2s.append(bsf)
        feat2 = []
        feat2.append(feats[self.num_levels-1])
        gather_size = feats[self.num_levels-1].size()[2:]
        gathered2 = F.adaptive_max_pool2d(feats[self.num_levels-2], output_size=gather_size)
        gathered2 = self.refine1[self.num_levels-1](gathered2)
        feat2.append(gathered2)
        bsf = sum(feat2) / len(feat2)
        bsf = self.refine3(bsf)
        feat2s.append(bsf)        
        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            # print("feats[i].size():",feats[i].size())
            # print("inputs[i].size():",inputs[i].size())
            outs.append(feat2s[i] + inputs[i])
        return tuple(outs)