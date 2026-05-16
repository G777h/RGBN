import time

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.ops import ste_round
from modules.transform import *
from utils.ckbd import *
from utils.moduleFunc import get_scale_table, update_registered_buffers


class SliceAwareGGDM(nn.Module):
   
    def __init__(self, slice_num, get_c_dim_fn, get_l_dim_fn, mid_ch=128):
        super().__init__()
        
        self.in_l = nn.ModuleList([nn.Conv2d(get_l_dim_fn(i), mid_ch, 1) for i in range(slice_num)])
        self.out_g = nn.ModuleList([nn.Conv2d(mid_ch, get_c_dim_fn(i), 1) for i in range(slice_num)])

        self.net = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch),
            nn.ReLU(True),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch),
            nn.ReLU(True)
        )

    def forward(self, c_feat, l_feat, idx):
        l_proj = self.in_l[idx](l_feat)               
        g_core = self.net(l_proj)                     
        g = torch.sigmoid(self.out_g[idx](g_core))     
        # [Fix Issue-5] 返回 gate 加权和补充两路特征，让调用处保留互补信息
        return c_feat * g, c_feat * (1 - g)


class SliceAwareGPF(nn.Module):
    
    def __init__(self, slice_num, get_m_dim_fn, get_a_dim_fn, mid_ch=128):
        super().__init__()
        
        self.in_m = nn.ModuleList([nn.Conv2d(get_m_dim_fn(i), mid_ch, 1) for i in range(slice_num)])
        self.in_a = nn.ModuleList([nn.Conv2d(get_a_dim_fn(i), mid_ch, 1) for i in range(slice_num)])
        self.out_m = nn.ModuleList([nn.Conv2d(mid_ch, get_m_dim_fn(i), 1) for i in range(slice_num)])

        self.adm = nn.Conv2d(mid_ch, mid_ch, 1)
        self.ada = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch),
            nn.LeakyReLU(0.2, True)
        )
        self.gate = nn.Sequential(
            # [Fix Issue-6] 去掉 groups 限制，改用普通卷积充分混合跨模态通道
            nn.Conv2d(mid_ch * 2, mid_ch, 3, 1, 1),  # 全通道卷积实现跨模态信息交流
            nn.ReLU(True),
            nn.Conv2d(mid_ch, mid_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, hm, ha, idx):
        m_proj = self.in_m[idx](hm)
        a_proj = self.in_a[idx](ha)

        mp, ap = self.adm(m_proj), self.ada(a_proj)
        core_out = mp + self.gate(torch.cat([mp, ap], 1)) * ap

        return self.out_m[idx](core_out)


class ELIC_united(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch
        self.quant = config.quant 
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEXcross(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEXcross(N, M, act=nn.ReLU)
        self.h_a = HyperAnalysisEXcross(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEXcross(N, M, act=nn.ReLU)

        self.rgb_local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        self.rgb_local_context_anchor_with_nonanchor = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        self.depth_local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )

        self.rgb_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        self.depth_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )

        def get_init_dim(i):
            return M * 4 + slice_ch[i] * 4 if i > 0 else M * 4

        def get_slice_ch_2(i):
            return slice_ch[i] * 2
            
        def get_slice_ch_4(i):
            return slice_ch[i] * 4

        # [Fix Issue-5] GGDM 返回两路特征（gate + residual 共 4ch）+ 跨模态 local_ctx (2ch) = 6ch
        def get_slice_ch_6(i):
            return slice_ch[i] * 6

        mid_ch = 128  

        self.depth_anchor_gpf = SliceAwareGPF(slice_num, get_init_dim, get_slice_ch_2, mid_ch=mid_ch)

        self.rgb_ggdm = SliceAwareGGDM(slice_num, get_slice_ch_2, get_slice_ch_2, mid_ch=mid_ch)
        # [Fix Issue-5] 辅助输入维度由 4ch 升为 6ch
        self.rgb_nonanchor_gpf = SliceAwareGPF(slice_num, get_init_dim, get_slice_ch_6, mid_ch=mid_ch)

        self.depth_ggdm = SliceAwareGGDM(slice_num, get_slice_ch_2, get_slice_ch_2, mid_ch=mid_ch)
        # [Fix Issue-5] 辅助输入维度由 4ch 升为 6ch
        self.depth_nonanchor_gpf = SliceAwareGPF(slice_num, get_init_dim, get_slice_ch_6, mid_ch=mid_ch)

        self.rgb_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=get_init_dim(i), out_dim=slice_ch[i] * 2, act=nn.ReLU) for i in range(slice_num)
        )
        self.depth_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=get_init_dim(i), out_dim=slice_ch[i] * 2, act=nn.ReLU) for i in range(slice_num)
        )
        self.rgb_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=get_init_dim(i), out_dim=slice_ch[i] * 2, act=nn.ReLU) for i in range(slice_num)
        )
        self.depth_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=get_init_dim(i), out_dim=slice_ch[i] * 2, act=nn.ReLU) for i in range(slice_num)
        )

        self.entropy_bottleneck = None
        self.rgb_entropy_bottleneck = EntropyBottleneck(N)
        self.depth_entropy_bottleneck = EntropyBottleneck(N)

        self.gaussianConditional = None
        self.rgb_gaussian_conditional = GaussianConditional(None)
        self.depth_gaussian_conditional = GaussianConditional(None)

    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def codeOnePart(self, slice_part, contextList, params_module, split_func, entropy_model, **kargs):
        params_one_part = params_module(torch.cat(contextList, 1))
        scales, means = params_one_part.chunk(2, 1)
        scales = split_func(scales)
        means = split_func(means)
        if self.quant == "ste":
            # [Fix Issue-7] ste 模式同样需要 split_func 应用空间掩码，保证 anchor/nonanchor 位置正确
            slice_part = ste_round(slice_part - means) + means
            slice_part = split_func(slice_part)  # 修复点：与 noise 模式保持一致
        else:
            slice_part = entropy_model.quantize(slice_part, "noise" if self.training else "dequantize")
            slice_part = split_func(slice_part)

        if "anchor_part" in kargs.keys():
            slices = slice_part + kargs["anchor_part"]
            if "local_context" in kargs.keys():
                localctx = kargs["local_context"](slices)
                return slice_part, scales, means, localctx, slices
            return slice_part, scales, means, slices

        if "local_context" in kargs.keys():
            localctx = kargs["local_context"](slice_part)
            return slice_part, scales, means, localctx
        return slice_part, scales, means

    def entropy_estimate_one_slice(
        self, rgb_y_slice, depth_y_slice, rgb_hyper_params, depth_hyper_params, rgb_y_hat_slices, depth_y_hat_slices, idx
    ):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)

        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = torch.cat([rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx], dim=1)
        else:
            init_context = torch.cat([rgb_hyper_params, depth_hyper_params], dim=1)

        ## 1. rgb anchor
        rgb_slice_anchor, rgb_scales_anchor, rgb_means_anchor, rgb_local_ctx = self.codeOnePart(
            rgb_slice_anchor, [init_context], self.rgb_entropy_parameters_anchor[idx],
            ckbd_anchor, self.rgb_gaussian_conditional, local_context=self.rgb_local_context[idx]
        )

        depth_anchor_fused_ctx = self.depth_anchor_gpf(init_context, rgb_local_ctx, idx)
        depth_slice_anchor, depth_scales_anchor, depth_means_anchor, depth_local_ctx = self.codeOnePart(
            depth_slice_anchor, [depth_anchor_fused_ctx], self.depth_entropy_parameters_anchor[idx],
            ckbd_anchor, self.depth_gaussian_conditional, local_context=self.depth_local_context[idx]
        )

        rgb_gated, rgb_residual = self.rgb_ggdm(rgb_local_ctx, depth_local_ctx, idx)
        # [Fix Issue-5] 拼接 gate加权、互补和跨模态三路信息，充分利用 GGDM 两路输出
        rgb_nonanchor_aux = torch.cat([rgb_gated, rgb_residual, depth_local_ctx], dim=1)
        rgb_nonanchor_fused_ctx = self.rgb_nonanchor_gpf(init_context, rgb_nonanchor_aux, idx)

        (rgb_slice_nonanchor, rgb_scales_nonanchor, rgb_means_nonanchor, rgb_local_ctx_nonanchor, rgb_y_hat_slice) = self.codeOnePart(
            rgb_slice_nonanchor, [rgb_nonanchor_fused_ctx], self.rgb_entropy_parameters_nonanchor[idx],
            ckbd_nonanchor, self.rgb_gaussian_conditional, anchor_part=rgb_slice_anchor, local_context=self.rgb_local_context_anchor_with_nonanchor[idx]
        )

        depth_gated, depth_residual = self.depth_ggdm(depth_local_ctx, rgb_local_ctx_nonanchor, idx)
        # [Fix Issue-5] 拼接 gate加权、互补和跨模态三路信息
        depth_nonanchor_aux = torch.cat([depth_gated, depth_residual, rgb_local_ctx_nonanchor], dim=1)
        depth_nonanchor_fused_ctx = self.depth_nonanchor_gpf(init_context, depth_nonanchor_aux, idx)

        (depth_slice_nonanchor, depth_scales_nonanchor, depth_means_nonanchor, depth_y_hat_slice) = self.codeOnePart(
            depth_slice_nonanchor, [depth_nonanchor_fused_ctx], self.depth_entropy_parameters_nonanchor[idx],
            ckbd_nonanchor, self.depth_gaussian_conditional, anchor_part=depth_slice_anchor
        )

        ## BPP
        rgb_scales_slice = ckbd_merge(rgb_scales_anchor, rgb_scales_nonanchor)
        rgb_means_slice = ckbd_merge(rgb_means_anchor, rgb_means_nonanchor)
        _, rgb_y_slice_likelihoods = self.rgb_gaussian_conditional(rgb_y_slice, rgb_scales_slice, rgb_means_slice)

        depth_scales_slice = ckbd_merge(depth_scales_anchor, depth_scales_nonanchor)
        depth_means_slice = ckbd_merge(depth_means_anchor, depth_means_nonanchor)
        _, depth_y_slice_likelihoods = self.depth_gaussian_conditional(depth_y_slice, depth_scales_slice, depth_means_slice)

        return rgb_y_hat_slice, depth_y_hat_slice, rgb_y_slice_likelihoods, depth_y_slice_likelihoods

    def entropy_estimate_united(self, rgb, depth, rgb_hyper_params, depth_hyper_params):
        rgb_y_slices = [
            rgb[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        rgb_y_hat_slices = []
        rgb_y_likelihoods = []

        depth_y_slices = [
            depth[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        depth_y_hat_slices = []
        depth_y_likelihoods = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            (
                rgb_y_hat_slice,
                depth_y_hat_slice,
                rgb_y_slice_likelihoods,
                depth_y_slice_likelihoods,
            ) = self.entropy_estimate_one_slice(
                rgb_y_slice,
                depth_y_slice,
                rgb_hyper_params,
                depth_hyper_params,
                rgb_y_hat_slices,
                depth_y_hat_slices,
                idx,
            )

            rgb_y_hat_slices.append(rgb_y_hat_slice)
            rgb_y_likelihoods.append(rgb_y_slice_likelihoods)
            depth_y_hat_slices.append(depth_y_hat_slice)
            depth_y_likelihoods.append(depth_y_slice_likelihoods)

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        rgb_y_likelihoods = torch.cat(rgb_y_likelihoods, dim=1)

        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)
        depth_y_likelihoods = torch.cat(depth_y_likelihoods, dim=1)

        return rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods

    def forward(self, rgb, depth):
        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验估计
        rgb_z_hat, rgb_z_likelihoods = self.rgb_entropy_bottleneck(rgb_z)
        depth_z_hat, depth_z_likelihoods = self.depth_entropy_bottleneck(depth_z)
        if self.quant == "ste":
            rgb_z_offset = self.rgb_entropy_bottleneck._get_medians()
            rgb_z_hat = ste_round(rgb_z - rgb_z_offset) + rgb_z_offset
            depth_z_offset = self.depth_entropy_bottleneck._get_medians()
            depth_z_hat = ste_round(depth_z - depth_z_offset) + depth_z_offset

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods = self.entropy_estimate_united(
            rgb_y, depth_y, rgb_hyper_params, depth_hyper_params
        )

        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        return {
            "x_hat": {"r": rgb_hat, "d": depth_hat},
            "r_likelihoods": {"y": rgb_y_likelihoods, "z": rgb_z_likelihoods},
            "d_likelihoods": {"y": depth_y_likelihoods, "z": depth_z_likelihoods},
        }

    def compress_one_slice(
        self,
        rgb_y_slice,
        depth_y_slice,
        rgb_hyper_params,
        depth_hyper_params,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        idx,
        rgb_symbols_list,
        rgb_indexes_list,
        depth_symbols_list,
        depth_indexes_list,
    ):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]
            
        init_context = torch.cat(init_context, dim=1)

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](init_context)
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        rgb_slice_anchor = compress_anchor(
            self.rgb_gaussian_conditional,
            rgb_slice_anchor,
            rgb_scales_anchor,
            rgb_means_anchor,
            rgb_symbols_list,
            rgb_indexes_list,
        )
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # depth-anchor
        depth_anchor_fused = self.depth_anchor_gpf(init_context, rgb_local_ctx, idx)
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](depth_anchor_fused)
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = compress_anchor(
            self.depth_gaussian_conditional,
            depth_slice_anchor,
            depth_scales_anchor,
            depth_means_anchor,
            depth_symbols_list,
            depth_indexes_list,
        )
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # rgb-nonanchor
        rgb_gated, rgb_residual = self.rgb_ggdm(rgb_local_ctx, depth_local_ctx, idx)
        # [Fix Issue-5] 拼接三路信息，与 training forward 保持一致
        rgb_nonanchor_fused = self.rgb_nonanchor_gpf(init_context, torch.cat([rgb_gated, rgb_residual, depth_local_ctx], dim=1), idx)
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](rgb_nonanchor_fused)
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        rgb_slice_nonanchor = compress_nonanchor(
            self.rgb_gaussian_conditional,
            rgb_slice_nonanchor,
            rgb_scales_nonanchor,
            rgb_means_nonanchor,
            rgb_symbols_list,
            rgb_indexes_list,
        )
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx_nonanchor = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)

        # depth-nonanchor
        depth_gated, depth_residual = self.depth_ggdm(depth_local_ctx, rgb_local_ctx_nonanchor, idx)
        # [Fix Issue-5] 拼接三路信息，与 training forward 保持一致
        depth_nonanchor_fused = self.depth_nonanchor_gpf(init_context,
                                                         torch.cat([depth_gated, depth_residual, rgb_local_ctx_nonanchor], dim=1), idx)
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](depth_nonanchor_fused)
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = compress_nonanchor(
            self.depth_gaussian_conditional,
            depth_slice_nonanchor,
            depth_scales_nonanchor,
            depth_means_nonanchor,
            depth_symbols_list,
            depth_indexes_list,
        )
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor

        rgb_y_hat_slices.append(rgb_y_hat_slice)
        depth_y_hat_slices.append(depth_y_hat_slice)
        return rgb_y_hat_slices, depth_y_hat_slices

    def compress_united(self, rgb_y, rgb_hyper_params, depth_y, depth_hyper_params):
        rgb_y_slices = [
            rgb_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        rgb_y_hat_slices = []

        depth_y_slices = [
            depth_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        depth_y_hat_slices = []

        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_encoder = BufferedRansEncoder()
        rgb_symbols_list = []
        rgb_indexes_list = []
        rgb_y_strings = []

        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_encoder = BufferedRansEncoder()
        depth_symbols_list = []
        depth_indexes_list = []
        depth_y_strings = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            rgb_y_hat_slices, depth_y_hat_slices = self.compress_one_slice(
                rgb_y_slice,
                depth_y_slice,
                rgb_hyper_params,
                depth_hyper_params,
                rgb_y_hat_slices,
                depth_y_hat_slices,
                idx,
                rgb_symbols_list,
                rgb_indexes_list,
                depth_symbols_list,
                depth_indexes_list,
            )

        rgb_encoder.encode_with_indexes(rgb_symbols_list, rgb_indexes_list, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_y_string = rgb_encoder.flush()
        rgb_y_strings.append(rgb_y_string)

        depth_encoder.encode_with_indexes(
            depth_symbols_list, depth_indexes_list, depth_cdf, depth_cdf_lengths, depth_offsets
        )
        depth_y_string = depth_encoder.flush()
        depth_y_strings.append(depth_y_string)
        return rgb_y_strings, depth_y_strings

    def compress(self, rgb, depth):
        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验
        torch.backends.cudnn.deterministic = True
        rgb_z_strings = self.rgb_entropy_bottleneck.compress(rgb_z)
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, rgb_z.size()[-2:])
        depth_z_strings = self.depth_entropy_bottleneck.compress(depth_z)
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, depth_z.size()[-2:])

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_strings, depth_y_strings = self.compress_united(rgb_y, rgb_hyper_params, depth_y, depth_hyper_params)

        torch.backends.cudnn.deterministic = False
        return {
            "r_strings": [rgb_y_strings, rgb_z_strings],
            "d_strings": [depth_y_strings, depth_z_strings],
            "shape": rgb_z.size()[-2:],
        }

    def decompress(self, rgb_strings, depth_strings, shape):
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        rgb_y_strings = rgb_strings[0][0]  
        rgb_z_strings = rgb_strings[1]
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, shape)
        depth_y_strings = depth_strings[0][0]
        depth_z_strings = depth_strings[1]
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, shape)

        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)

        rgb_y_hat, depth_y_hat = self.decompress_united(
            rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params
        )
        torch.backends.cudnn.deterministic = False
        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time
        return {"x_hat": {"r": rgb_hat.clamp_(0, 1), "d": depth_hat.clamp_(0, 1)}, "cost_time": cost_time}

    def decompress_one_slice(
        self,
        rgb_decoder,
        depth_decoder,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        rgb_hyper_params,
        depth_hyper_params,
        idx,
        rgb_cdf,
        rgb_cdf_lengths,
        rgb_offsets,
        depth_cdf,
        depth_cdf_lengths,
        depth_offsets,
    ):
        init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        init_context = torch.cat(init_context, dim=1)

        # 2. rgb-anchor 解码
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](init_context) 
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        rgb_slice_anchor = decompress_anchor(
            self.rgb_gaussian_conditional,
            rgb_scales_anchor,
            rgb_means_anchor,
            rgb_decoder,
            rgb_cdf,
            rgb_cdf_lengths,
            rgb_offsets,
        )
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # 3. depth-anchor 解码
        depth_anchor_fused = self.depth_anchor_gpf(init_context, rgb_local_ctx, idx)
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](depth_anchor_fused)
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = decompress_anchor(
            self.depth_gaussian_conditional,
            depth_scales_anchor,
            depth_means_anchor,
            depth_decoder,
            depth_cdf,
            depth_cdf_lengths,
            depth_offsets,
        )
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # 4. rgb-nonanchor 解码
        rgb_gated, rgb_residual = self.rgb_ggdm(rgb_local_ctx, depth_local_ctx, idx)
        # [Fix Issue-5] 拼接三路信息，与 training forward 保持一致
        rgb_nonanchor_fused = self.rgb_nonanchor_gpf(init_context, torch.cat([rgb_gated, rgb_residual, depth_local_ctx], dim=1), idx)
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](rgb_nonanchor_fused)
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        rgb_slice_nonanchor = decompress_nonanchor(
            self.rgb_gaussian_conditional,
            rgb_scales_nonanchor,
            rgb_means_nonanchor,
            rgb_decoder,
            rgb_cdf,
            rgb_cdf_lengths,
            rgb_offsets,
        )
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx_nonanchor = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)
        rgb_y_hat_slices.append(rgb_y_hat_slice)

        # 5. depth-nonanchor 解码
        depth_gated, depth_residual = self.depth_ggdm(depth_local_ctx, rgb_local_ctx_nonanchor, idx)
        # [Fix Issue-5] 拼接三路信息，与 training forward 保持一致
        depth_nonanchor_fused = self.depth_nonanchor_gpf(init_context, torch.cat([depth_gated, depth_residual, rgb_local_ctx_nonanchor], dim=1), idx)
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](depth_nonanchor_fused)
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = decompress_nonanchor(
            self.depth_gaussian_conditional,
            depth_scales_nonanchor,
            depth_means_nonanchor,
            depth_decoder,
            depth_cdf,
            depth_cdf_lengths,
            depth_offsets,
        )
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor
        depth_y_hat_slices.append(depth_y_hat_slice)

        return rgb_y_hat_slices, depth_y_hat_slices

    def decompress_united(self, rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params):
        rgb_y_hat_slices = []
        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_decoder = RansDecoder()
        rgb_decoder.set_stream(rgb_y_strings)

        depth_y_hat_slices = []
        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_decoder = RansDecoder()
        depth_decoder.set_stream(depth_y_strings)

        for idx in range(self.slice_num):
            rgb_y_hat_slices, depth_y_hat_slices = self.decompress_one_slice(
                rgb_decoder,
                depth_decoder,
                rgb_y_hat_slices,
                depth_y_hat_slices,
                rgb_hyper_params,
                depth_hyper_params,
                idx,
                rgb_cdf,
                rgb_cdf_lengths,
                rgb_offsets,
                depth_cdf,
                depth_cdf_lengths,
                depth_offsets,
            )

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)

        return rgb_y_hat, depth_y_hat

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        rgb_updated = self.rgb_gaussian_conditional.update_scale_table(scale_table, force=force)
        depth_updated = self.depth_gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = rgb_updated & depth_updated | super().update(force=force)  
        return updated

    def load_state_dict(self, state_dict, strict=False):
        update_registered_buffers(
            self.rgb_gaussian_conditional,
            "rgb_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.depth_gaussian_conditional,
            "depth_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.rgb_entropy_bottleneck,
            "rgb_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.depth_entropy_bottleneck,
            "depth_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        
        result = nn.Module.load_state_dict(self, state_dict, strict=strict)
        if strict:
           print("ELIC_united load state dict strict=True success.")
        else:
           print("ELIC_united load state dict strict=False success (missing keys allowed).")

        if result is None:
           print("ERROR: nn.Module.load_state_dict just returned None! This should never happen.")
           raise RuntimeError("nn.Module.load_state_dict returned None")
        print(f"Type of result: {type(result)}, missing_keys count: {len(result.missing_keys)}, unexpected_keys count: {len(result.unexpected_keys)}")
        return result
