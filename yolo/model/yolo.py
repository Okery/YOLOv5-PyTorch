import math

from torch import nn

from .head import Head
from .backbone import darknet_pan_backbone
from .transform import Transformer


class YOLOv5(nn.Module):
    def __init__(self, num_classes, model_size=(0.33, 0.5),
                 match_thresh=4, giou_ratio=1, img_sizes=(320, 416),
                 score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        # original
        anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        # [320, 416]
        anchors2 = [
            [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
            [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
            [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
        ]
        loss_weights = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.58}
        
        self.backbone = darknet_pan_backbone(
            depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        
        in_channels_list = self.backbone.body.out_channels_list
        strides = (8, 16, 32)
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, strides)
        
        self.head = Head(
            predictor, anchors, strides, 
            match_thresh, giou_ratio, loss_weights, 
            score_thresh, nms_thresh, detections)
        
        if isinstance(img_sizes, int):
            img_sizes = (img_sizes, img_sizes)
        self.transformer = Transformer(
            min_size=img_sizes[0], max_size=img_sizes[1], 
            image_mean=(0.485, 0.456, 0.406), 
            image_std=(0.229, 0.224, 0.225))
    
    def forward(self, images, targets=None):
        orig_image_shapes = [img.shape[1:] for img in images]

        images, targets, image_shapes = self.transformer(images, targets)
        batch_shape = images.shape[2:]
        
        features = self.backbone(images)
        
        results, losses = self.head(features, batch_shape, targets, image_shapes)
        
        if self.training:
            return losses
        else:
            results = self.transformer.postprocess(results, image_shapes, orig_image_shapes)
            return results, losses
        
    def fuse(self):
        # fusing conv and bn layers
        # However, this method has no effect for accelerating inferencing.
        # Perhaps PyTorch has fused conv and bn layers automatically.
        for m in self.modules():
            if hasattr(m, "fused"):
                m.fuse()


class Predictor(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, strides):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
            
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        for m, n, s in zip(self.mlp, num_anchors, strides):
            b = m.bias.view(n, -1)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (num_classes - 0.99))
            m.bias = nn.Parameter(b.view(-1), requires_grad=True)
            
    def forward(self, x):
        N = x[0].shape[0]
        L = self.num_outputs
        preds = []
        for i in range(len(x)):
            h, w = x[i].shape[-2:]
            pred = self.mlp[i](x[i])
            pred = pred.permute(0, 2, 3, 1).reshape(N, h, w, -1, L)
            preds.append(pred)
        return preds
    
    