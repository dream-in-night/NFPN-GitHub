import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS
class FPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 backbone = None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.backbone = None
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # print("=======================")
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@NECKS.register_module()
class NeiFPN(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
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
                 out_channels,
                 num_levels,
                 num_outs,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 backbone = None):
        super(NeiFPN, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # self.backbone = None
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        self.to_reslayer = nn.ModuleList()
        for out_channel in [64,256,512,1024]:
            d_conv = ConvModule(
                self.in_channels,
                out_channel,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            self.to_reslayer.append(d_conv)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()# 64 256 512 1024
        self.fpn = FPN()
        for i in range(5):
            d_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
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

    def forward(self, inputs, context):
        """Forward function."""
        inputs = self.fpn(inputs)
        assert len(inputs) == self.num_levels
        gather_sizes = []
        for i in range(len(inputs)):
            gather_size = inputs[i].size()[2:]
            gather_sizes.append(gather_size)
        feats = []
        feat = []
        feat.append(inputs[0])
        gather_size = inputs[0].size()[2:]
        gathered2 = F.interpolate(inputs[1], size=gather_size, mode='bilinear')
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
            gathered2 = F.interpolate(inputs[i+1], size=gather_size, mode='bilinear')
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
        outs = []
        for i in range(self.num_levels):
            # print("feats[i].size():",feats[i].size())
            # print("inputs[i].size():",inputs[i].size())
            outs.append(feats[i] + inputs[i])
        gathered0, gathered1, gathered2, gathered3, gathered4 = outs[0], outs[1], outs[2], outs[3], outs[4], 
        res_layers = []
        for i, layer_name in enumerate(context.res_layers):
            res_layer = getattr(context, layer_name)
            res_layers.append(res_layer)
        # level 0 256->64->256->256 s=1
        out_0 = gathered0 + inputs[0] + self.fpn.lateral_convs[0](res_layers[0](self.to_reslayer[0](gathered0))) + self.refine(F.interpolate(gathered2, size=gather_sizes[0], mode='bilinear'))
        # out_0 = gathered0 + inputs[0] + self.refine(F.interpolate(gathered2, size=gather_sizes[0], mode='bilinear'))
        # level 1 256->256->512->256 s=2
        # print("gathered0:",gathered0.size())
        out_1 = gathered1 + self.fpn.lateral_convs[1](res_layers[1](self.to_reslayer[1](gathered0))) + inputs[1] + self.refine(F.interpolate(gathered2, size=gather_sizes[1], mode='bilinear'))
        # out_1 = gathered1 + inputs[1] + self.refine(F.interpolate(gathered2, size=gather_sizes[1], mode='nearest'))
        # level 2 256->512->1024->256 s=2
        out_2 = gathered2 + self.fpn.lateral_convs[2](res_layers[2](self.to_reslayer[2](gathered1))) + inputs[2] 
        # out_2 = gathered2 + inputs[2] 
        # level 3 256->1024->2048->256 s=2
        out_3 = gathered3 + self.fpn.lateral_convs[3](res_layers[3](self.to_reslayer[3](gathered2))) + inputs[3] + F.adaptive_max_pool2d(gathered2, output_size=gather_sizes[3])
        # out_3 = gathered3 + inputs[3] + F.adaptive_max_pool2d(gathered2, output_size=gather_sizes[3])
        # level 3
        out_4 = gathered4 + inputs[4] + F.adaptive_max_pool2d(gathered2, output_size=gather_sizes[4])        
       
        # bsf = self.refine(bsf)
        # step 3: scatter refined features to multi-levels by a residual path
        outs = [out_0,out_1,out_2,out_3,out_4]
        return [tuple(inputs),tuple(outs)]
