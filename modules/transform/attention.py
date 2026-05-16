import torch
from torch import nn
from torch.nn import functional as F
import torchvision.ops as ops

from .spatialAligner import Spatial_aligner


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class bi_spf_single(nn.Module):
    def __init__(self, N):
        super(bi_spf_single, self).__init__()
        self.r_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.r_act = nn.ReLU()

        self.d_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.d_act = nn.ReLU()
        self.d_esa = ESA(N)

    def forward(self, rgb, depth):
        rgb = self.r_ext(rgb)
        rgb = self.r_act(rgb)
        depth = self.d_ext(depth)
        depth = self.d_act(depth)

        d = self.d_esa(torch.cat((depth, rgb), dim=-3))
        return d


class bi_spf(bi_spf_single):
    def __init__(self, N):
        super().__init__(N)
        self.r_esa = ESA(N)

    def forward(self, rgb, depth):
        rgb = self.r_ext(rgb)
        rgb = self.r_act(rgb)
        depth = self.d_ext(depth)
        depth = self.d_act(depth)

        r = self.r_esa(torch.cat((rgb, depth), dim=-3))
        d = self.d_esa(torch.cat((depth, rgb), dim=-3))
        return r, d


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = x
        c1_ = self.conv1(f)  # 1*1卷积，降低维度（减少计算复杂度）
        c1 = self.conv2(c1_)  # 减小特征图尺寸
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 减小特征图尺寸，增大感受野
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)  # 上采样，恢复特征图尺寸
        cf = self.conv_f(c1_)  #
        c4 = self.conv4(c3 + cf)  # 1*1卷积恢复通道数
        m = self.sigmoid(c4)  # 生成mask

        return x * m

class CMGM(nn.Module):
    """
    Cross-Modal Multi-scale Gating Module (单向辅助：RGB -> Normal)
    实现方案 A: G = Sigmoid( sum(DWConv_i(F_n)) )
    """
    def __init__(self, in_channels):
        super().__init__()
        # 1. 多尺度空间上下文提取 (使用 groups=in_channels 实现 DWConv 深度可分离卷积)
        # Scale 1: 感受野 3x3 (dilation=1)
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, groups=in_channels)
        # Scale 2: 感受野 5x5 (dilation=2)
        self.dwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        # Scale 3: 感受野 7x7 (dilation=3)
        self.dwconv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels)
        
        # 2. 零初始化卷积 (冷启动保护)
        self.zero_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, rgb_feat, norm_feat):
        # 1. Target (Normal) 分支生成多尺度感知特征
        g1 = self.dwconv1(norm_feat)
        g2 = self.dwconv2(norm_feat)
        g3 = self.dwconv3(norm_feat)
        
        # 2. 方案 A: 融合特征后计算 Sigmoid 生成跨模态门控 (Mask)
        gate = torch.sigmoid(g1 + g2 + g3)
        
        # 3. 门控筛选 Source (RGB) 分支，严格压制与几何无关的纹理噪声
        modulated_rgb = rgb_feat * gate
        
        # 4. 零卷积输出最终的特征增量 (Delta)
        norm_delta = self.zero_conv(modulated_rgb)
        
        return norm_delta
