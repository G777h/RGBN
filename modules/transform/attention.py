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

    # 仅仅辅助第二个
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


# rgb和depth根据空间均值来进行通道加权【通过全局池化来实现】
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
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

class CMGM_v3_core(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        mid_channels = max(8, in_channels // reduction_ratio)

        # 表面对齐感知预测器 (Offset Predictor)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 18, 3, padding=1)
        )
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

        self.deformable_conv = ops.DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.source_compress = nn.Conv2d(in_channels, mid_channels, 1)
        self.target_compress = nn.Conv2d(in_channels, mid_channels, 1)

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(mid_channels * 2, in_channels, 1),
            nn.GELU()
        )

        self.output_proj = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def compute_physical_cosine_gate(self, target_feat):
        norm_vec = F.normalize(target_feat, p=2, dim=1)
        sim_x = F.cosine_similarity(norm_vec[:, :, :, :-1], norm_vec[:, :, :, 1:], dim=1)
        sim_y = F.cosine_similarity(norm_vec[:, :, :-1, :], norm_vec[:, :, 1:, :], dim=1)
        sim_x = F.pad(sim_x, (0, 1, 0, 0), value=1.0)
        sim_y = F.pad(sim_y, (0, 0, 0, 1), value=1.0)
        physical_gate = F.relu((sim_x + sim_y) / 2.0).unsqueeze(1)
        return physical_gate

    def forward(self, source_feat, target_feat):
        # 漏斗梯度先验
        source_prior = source_feat.detach() * 0.9 + source_feat * 0.1

        # 表面对齐感知与可变形卷积
        offsets = self.offset_predictor(target_feat)
        aligned_source = self.deformable_conv(source_prior, offsets)

        # 物理门控截断
        physical_gate = self.compute_physical_cosine_gate(target_feat)
        gated_aligned_source = aligned_source * physical_gate

        # 特征压缩与融合
        source_comp = self.source_compress(gated_aligned_source)
        target_comp = self.target_compress(target_feat)
        fused = self.fusion_proj(torch.cat([source_comp, target_comp], dim=1))

        output = self.output_proj(fused)
        # 注意：这里直接返回 delta 特征，交给外部的 torch.cat 进行拼接
        return self.alpha * output


class Bi_CMGM_v3(nn.Module):
    """双向封装：用于完美替换原代码的 bi_spf"""

    def __init__(self, in_channels):
        super().__init__()
        # RGB 作为 source 引导 Normal (即 Normal 吸取 RGB 特征)
        self.rgb_to_norm = CMGM_v3_core(in_channels)
        # Normal 作为 source 引导 RGB (即 RGB 吸取 Normal 特征)
        self.norm_to_rgb = CMGM_v3_core(in_channels)

    def forward(self, rgb_y, depth_y):
        # 输出的分别为 RGB分支 和 Normal分支 的增量特征
        norm_f = self.rgb_to_norm(source_feat=rgb_y, target_feat=depth_y)
        rgb_f = self.norm_to_rgb(source_feat=depth_y, target_feat=rgb_y)
        return rgb_f, norm_f
