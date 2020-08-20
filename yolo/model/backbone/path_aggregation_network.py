import torch
from torch import nn

from .utils import Conv, ConcatBlock


class PathAggregationNetwork(nn.Module):
    def __init__(self, in_channels_list, depth):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        self.downsample_blocks = nn.ModuleList()
        self.outer_blocks = nn.ModuleList()
        
        for i, ch in enumerate(in_channels_list):
            self.inner_blocks.append(ConcatBlock(2 * ch if i < 2 else in_channels_list[-1], ch, depth, False))
            if i > 0:
                in_channels = in_channels_list[i - 1]
                self.layer_blocks.append(Conv(ch, in_channels, 1))
                self.upsample_blocks.append(nn.Upsample(scale_factor=2))
                
                self.downsample_blocks.append(Conv(in_channels, in_channels, 3, 2))
                self.outer_blocks.append(ConcatBlock(ch, ch, depth, False))
                
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_uniform_(m.weight, a=1)
        
    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for i in range(len(x) - 2, -1, -1):
            inner_top_down = self.upsample_blocks[i](results[0])
            last_inner = self.inner_blocks[i](torch.cat((inner_top_down, x[i]), dim=1)) # official
            #last_inner = self.inner_blocks[i](torch.cat((x[i], inner_top_down), dim=1)) # old
            results.insert(0, last_inner if i == 0 else self.layer_blocks[i - 1](last_inner))
            
        for i in range(len(x) - 1):
            outer_bottom_up = self.downsample_blocks[i](results[i])
            layer_result = results[i + 1]
            results[i + 1] = self.outer_blocks[i](torch.cat((outer_bottom_up, layer_result), dim=1))
            
        return results

    