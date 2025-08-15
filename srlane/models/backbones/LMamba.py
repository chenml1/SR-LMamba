import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from srlane.models.backbones.lib_mamba.vmambanew import SS2D
from functools import partial
from timm.layers import DropPath
from torch.nn.modules.batchnorm import _BatchNorm
from srlane.models.registry import BACKBONES


def gabor_filter(theta, size=5, sigma=0.5, gamma=0.75, lambd=1.0):

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
  
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    

    gb_real = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lambd)
    gb_imag = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.sin(2 * np.pi * x_theta / lambd)
    

    gabor = np.sqrt(gb_real**2 + gb_imag**2)
    

    gabor = gabor / np.sum(np.abs(gabor))
    return torch.tensor(gabor, dtype=torch.float32)

def create_curvelet_filter(num_angles, kernel_size=5, in_size=1, out_size=1):

    angles = [i * np.pi / num_angles for i in range(num_angles)]
    filters = []
    for theta in angles:
        filter_ = gabor_filter(theta, kernel_size)
        filters.append(filter_)

    dec_filters = torch.stack(filters, dim=0)  # [num_angles, size, size]
    dec_filters = dec_filters.unsqueeze(1).repeat(1, in_size, 1, 1)  # [num_angles, in_size, size, size]
    
    rec_filters = dec_filters.clone() 
    
    return dec_filters, rec_filters

def curvelet_transform(x, filters):
    B, C, H, W = x.shape
    num_angles = filters.shape[0]
    device = x.device
    filters = filters.to(device)
    filter_bank = filters.reshape(-1, 1, filters.shape[-2], filters.shape[-1])

    x_expanded = x.repeat(1, num_angles, 1, 1)  # [B, C*num_angles, H, W]

    pad_h = (filters.shape[-2] - 1) // 2
    pad_w = (filters.shape[-1] - 1) // 2
    x_conv = F.conv2d(
        x_expanded, 
        filter_bank, 
        padding=(pad_h, pad_w),
        groups=C * num_angles
    )
    return x_conv

def inverse_curvelet_transform(x, filters):
    B, C_total, H, W = x.shape
    num_angles = filters.shape[0]
    C = C_total // num_angles
    filter_bank = filters.reshape(-1, 1, filters.shape[-2], filters.shape[-1])  # [num_angles*C, 1, k, k]
    pad = (filters.shape[-2] // 2, filters.shape[-1] // 2)
    x = F.conv_transpose2d(x, filter_bank, padding=pad, groups=C*num_angles)
    x = x.reshape(B, num_angles, C, H, W).sum(dim=1) / num_angles
    return x

class MBCVTConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, 
                 bias=True, cv_levels=2, num_angles=12, ssm_ratio=1, forward_type="v05"):
        super(MBCVTConv2d, self).__init__()
        assert in_channels == out_channels, 
        self.in_channels = in_channels
        self.cv_levels = cv_levels
        self.num_angles = num_angles
        self.stride = stride

        self.dec_filters, self.rec_filters = create_curvelet_filter(
            num_angles=num_angles,
            kernel_size=kernel_size,
            in_size=in_channels,
            out_size=out_channels
        )
        self.register_buffer('dec_filters_tensor', self.dec_filters)
        self.register_buffer('rec_filters_tensor', self.rec_filters)
        self.curvelet_convs = nn.ModuleList()
        self.curvelet_scale = nn.ModuleList()
        for i in range(cv_levels):
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels * num_angles,
                    in_channels * num_angles,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels * num_angles,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * num_angles),
                nn.ReLU(inplace=True)
            )
            self.curvelet_convs.append(conv_block)
            self.curvelet_scale.append(_ScaleModule([1, in_channels * num_angles, 1, 1], init_scale=0.1))

        self.global_atten = nn.Sequential(
            SS2D(
                d_model=in_channels, 
                d_state=1,
                ssm_ratio=ssm_ratio, 
                initialize="v2", 
                forward_type=forward_type, 
                channel_first=True, 
                k_group=2
            ),
            nn.BatchNorm2d(in_channels),
            _ScaleModule([1, in_channels, 1, 1])
        )
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = None

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x_orig = x
        for conv, scale in zip(self.curvelet_convs, self.curvelet_scale):
   
            dec_filters = self.dec_filters_tensor.to(x.device, non_blocking=True)
            rec_filters = self.rec_filters_tensor.to(x.device, non_blocking=True)

            x = curvelet_transform(x, dec_filters)
            x = conv(x)
            x = scale(x)
            x = inverse_curvelet_transform(x, rec_filters)
        x_cv = x + x_orig
        x_global = self.global_atten(x_orig)
        alpha = torch.sigmoid(self.fusion_weight)
        x_fused = alpha * x_global + (1 - alpha) * x_cv

        if self.downsample is not None:
            x_fused = self.downsample(x_fused)
            
        return x_fused.to(input_dtype)
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        try:
  
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            feat = torch.cat([max_out, avg_out], dim=1)
            return self.conv(feat)
        except Exception as e:
            print(f"Spatial attention error: {e}")
            return torch.ones_like(x[:, :1, :, :])

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        hidden_size = max(1, in_channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
      
        out = avg_out + max_out
        return out.view(b, c, 1, 1)

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class LMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=[0.8, 0.75, 0.7], local_ratio=[0.2, 0.25, 0.3],
                 kernels=7, ssm_ratio=1, forward_type="v052d", num_angles=12):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBCVTConv2d(
                self.global_channels, self.global_channels, 
                kernels, cv_levels=1, num_angles=num_angles,
                ssm_ratio=ssm_ratio, forward_type=forward_type)
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0,))

    def forward(self, x):  # x (B,C,H,W)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x


class LMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=7, ssm_ratio=1, forward_type="v052d", num_angles=12):
        super().__init__()
        self.dim = dim
        self.attn = LMambaModule(
            dim, 
            global_ratio=global_ratio, 
            local_ratio=local_ratio,
            kernels=kernels, 
            ssm_ratio=ssm_ratio, 
            forward_type=forward_type,
            num_angles=num_angles)

    def forward(self, x):
        x = self.attn(x)
        return x


class LMambaBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=7,  drop_path=0.1, has_skip=True, ssm_ratio=1, 
                 forward_type="v052d", num_angles=12):
        super().__init__()
        # 已删除refinement属性定义
        
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(LMambaBlockWindow(
                ed, 
                global_ratio=global_ratio, 
                local_ratio=local_ratio,
                kernels=kernels, 
                ssm_ratio=ssm_ratio,
                forward_type=forward_type,
                num_angles=num_angles))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        
        # 已删除特征精炼步骤
        
        return x

@BACKBONES.register_module
class LMamba(nn.Module):
    def __init__(self,
                 img_size=(320, 800),
                 in_chans=3,
                 stages=['s', 's', 's'],
                 embed_dim=[128, 256, 512],  # 👈 剪枝后的通道
                 global_ratio=[0.7, 0.6, 0.5],
                 local_ratio=[0.3, 0.4, 0.5],
                 depth=[3, 4, 3],
                 kernels=[7, 7, 7],
                 ssm_ratio=1,
                 forward_type="v052d",
                 out_indices=(0, 1, 2),
                 num_angles=8,  # 👈 减少方向数
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False, drop_path=0.1,
                 sync_bn=False, pretrained=None,
                 frozen_stages=-1, norm_eval=False, **kwargs):
        super().__init__()

        self.out_indices = out_indices

        # Initial downsampling
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1),
            nn.ReLU(),
        )

        # Stage 1
        self.downsample1 = nn.Sequential(
            Conv2d_BN(embed_dim[0], embed_dim[0], 3, 2, 1, groups=embed_dim[0]),
            nn.ReLU(),
        )
        self.blocks1 = nn.Sequential(*[
            LMambaBlock(
                stages[0], embed_dim[0], global_ratio[0], local_ratio[0],
                kernels[0], drop_path, ssm_ratio=ssm_ratio,
                forward_type=forward_type, num_angles=num_angles
            ) for _ in range(depth[0])
        ])

        # Stage 2
        self.downsample2 = nn.Sequential(
            Conv2d_BN(embed_dim[0], embed_dim[1], 1, 1, 0),
            nn.Conv2d(embed_dim[1], embed_dim[1], 3, 2, 1, groups=embed_dim[1], bias=False),
            nn.BatchNorm2d(embed_dim[1]),
            nn.ReLU(),
        )
        self.blocks2 = nn.Sequential(*[
            LMambaBlock(
                stages[1], embed_dim[1], global_ratio[1], local_ratio[1],
                kernels[1], drop_path, ssm_ratio=ssm_ratio,
                forward_type=forward_type, num_angles=num_angles
            ) for _ in range(depth[1])
        ])

        # Stage 3
        self.downsample3 = nn.Sequential(
            Conv2d_BN(embed_dim[1], embed_dim[2], 1, 1, 0),
            nn.Conv2d(embed_dim[2], embed_dim[2], 3, 2, 1, groups=embed_dim[2], bias=False),
            nn.BatchNorm2d(embed_dim[2]),
            nn.ReLU(),
        )
        self.blocks3 = nn.Sequential(*[
            LMambaBlock(
                stages[2], embed_dim[2], global_ratio[2], local_ratio[2],
                kernels[2], drop_path, ssm_ratio=ssm_ratio,
                forward_type=forward_type, num_angles=num_angles
            ) for _ in range(depth[2])
        ])

    def forward(self, x):
        out = []
        x = self.patch_embed(x)
        x = self.downsample1(x)
        x = self.blocks1(x)
        out.append(x)

        x = self.downsample2(x)
        x = self.blocks2(x)
        out.append(x)

        x = self.downsample3(x)
        x = self.blocks3(x)
        out.append(x)

        return tuple(out[i] for i in self.out_indices)
