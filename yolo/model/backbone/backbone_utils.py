from .path_aggregation_network import PathAggregationNetwork
from .utils import IntermediateLayerGetter
from .darknet import CSPDarknet

from torch import nn
    
    
class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.body = backbone
        self.fpn = fpn
        
    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    
    
def darknet_pan_backbone(depth_multiple, width_multiple): # 33 47 61 75
    out_channels_list = [round(width_multiple * x) for x in [64, 128, 256, 512, 1024]]
    layers = [max(round(depth_multiple * x), 1) for x in [3, 9, 9]]
    model = CSPDarknet(out_channels_list, layers) 

    return_layers = {'layer2', 'layer3', 'layer4'}

    backbone = IntermediateLayerGetter(model, return_layers)
    backbone.out_channels_list = out_channels_list[2:]
    
    depth = max(round(3 * depth_multiple), 1)
    fpn = PathAggregationNetwork(out_channels_list[2:], depth)
    return BackboneWithFPN(backbone, fpn)

