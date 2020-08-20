import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


bn_eps = 0.0001
bn_momentum = 0.03

class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))
    
    
class Mish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)
    
    
def fuse_conv_and_bn(conv, bn):
    conv_w, conv_b = conv.weight, conv.bias
    bn_w, bn_b = bn.weight, bn.bias
    bn_rm, bn_rv, bn_eps = bn.running_mean, bn.running_var, bn.eps

    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    fconv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(-1, 1, 1, 1)
    fconv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    fused_conv = copy.deepcopy(conv)
    fused_conv.weight = nn.Parameter(fconv_w)
    fused_conv.bias = nn.Parameter(fconv_b)
    return fused_conv
    
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, acti="leaky"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        if acti == "relu":
            self.acti = nn.ReLU(inplace=True)
        elif acti == "leaky":
            self.acti = nn.LeakyReLU(0.1, inplace=True)
        elif acti == "mish":
            self.acti = Mish()
            
        self.fused = False
            
    def forward(self, x):
        if not self.training and self.fused:
            return self.acti(self.fused_conv[0](x))
        else :
            return self.acti(self.bn(self.conv(x)))
            
    def fuse(self):
        self.fused = True
        # () is avoiding registering mechanism
        self.fused_conv = (fuse_conv_and_bn(self.conv, self.bn),)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fusion):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(out_channels, out_channels, 3)
        self.fusion = fusion and in_channels == out_channels
    
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.fusion:
            out = out + x
        return out


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer, fusion):
        super().__init__()
        mid_channels = out_channels // 2
        self.part1 = nn.Sequential(
            Conv(in_channels, mid_channels, 1),
            nn.Sequential(*[FusionBlock(mid_channels, mid_channels, fusion) for _ in range(layer)]),
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
        )
        self.part2 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.tail = nn.Sequential(
            nn.BatchNorm2d(2 * mid_channels, eps=bn_eps, momentum=bn_momentum),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.conv = Conv(2 * mid_channels, out_channels, 1)
        
    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        out = self.conv(self.tail(torch.cat((x1, x2), dim=1)))
        return out
    
    
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not return_layers.issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
            
        self.return_layers = return_layers
        layers = OrderedDict()
        n = 0
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                n += 1
            if n == len(return_layers):
                break

        super().__init__(layers)

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)
        return outputs
    

# attention modules

class ConvBlockAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.cam = ChannelAttention(in_channels, ratio)
        self.sam = SpatialAttention(kernel_size)
        
    def forward(self, x):
        Mc = self.cam(x)
        x = x * Mc
        Ms = self.sam(x)
        x = x * Ms
        return x
    
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        Mc = torch.sigmoid(avg_out + max_out)
        return Mc # (Tensor[N, C, 1, 1])


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = tuple((k - 1) // 2 for k in kernel_size)

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat((avg_out, max_out), dim=1)
        Ms = torch.sigmoid(self.conv(out))
        return Ms # (Tensor[N, 1, H, W])
    
    