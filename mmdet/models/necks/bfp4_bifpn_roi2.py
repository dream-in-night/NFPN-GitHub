import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d
import torch
from ..builder import NECKS
eps=0.0001

@NECKS.register_module()
class BFP_BIFP(nn.Module):
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
        super(BFP_BIFP, self).__init__()
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
        self.bifpn = BiFPNModule(channels=in_channels,conv_cfg=conv_cfg,norm_cfg=norm_cfg)   
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
        outs = []
        for i in range(self.num_levels):
            # print("feats[i].size():",feats[i].size())
            # print("inputs[i].size():",inputs[i].size())
            outs.append(feats[i] + inputs[i])
        bi_out = self.bifpn(inputs)
        return [tuple(inputs),tuple(outs),tuple(bi_out)]
class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels=5,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.levels = levels
        self.scale_convs = nn.ModuleList()
        for i in range(2):
            scale_conv = nn.Sequential(
                ConvModule(
                    channels,
                    channels//4,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    inplace=False),
                ConvModule(
                    channels//4,
                    5,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=False))
            self.scale_convs.append(scale_conv)
        #weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        self.bifpn_convs = nn.ModuleList()
        for jj in range(2):
            for i in range(self.levels-1):  # 0,1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        groups=channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=None,
                        inplace=False),
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        inplace=False))
                self.bifpn_convs.append(fpn_conv)
        self.linear = nn.Sequential(
            nn.Linear(200, 47),
            nn.ReLU(),
            nn.Linear(47, 5),
            nn.Softmax()
            )
    def forward(self, inputs):
        # print("len(inputs):",len(inputs))
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + eps #normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + eps
        # build top-down
        kk=0
        # pathtd = inputs copy is wrong
        pathtd=[inputs[levels - 1]]
#        for in_tensor in inputs:
#            pathtd.append(in_tensor.clone().detach())
        for i in range(levels - 1, 0, -1):
            t1 = w1[0,kk]*inputs[i - 1]
            a,b,w,h = t1.size()
            _t = t1 + w1[1,kk]*F.interpolate(
                pathtd[-1], size=(w,h), mode='nearest')
            pathtd.append(self.bifpn_convs[kk](_t))
            del(_t)
            kk=kk+1
        jj=kk
        pathtd = pathtd[::-1]#倒过来的意思
        # build down-top
        for i in range(0, levels - 2, 1):
            t1_ = w2[0, i] * inputs[i + 1]
            _,_,w,h = t1_.size()  
            pathtd[i + 1] = t1_ + w2[1, i] * nn.Upsample(size=(w,h))(pathtd[i]) + w2[2, i] * \
                            pathtd[i + 1]
            pathtd[i + 1] = self.bifpn_convs[jj](pathtd[i + 1])
            jj=jj+1
        t1_ = w1[0, kk] * inputs[levels - 1]
        _,_,w,h = t1_.size()        
        pathtd[levels - 1] = t1_ + w1[1, kk] * nn.Upsample(size=(w,h))(pathtd[levels - 2])
        pathtd[levels - 1] = self.bifpn_convs[jj](pathtd[levels - 1])
        for i in range(0, levels , 1):
            # pathtd[i] += scores_5[i]*3*inputs[i]
            pathtd[i] += inputs[i]
        return pathtd

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
