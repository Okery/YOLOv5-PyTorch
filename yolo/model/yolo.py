from torch import nn

from .head import Head
from .backbone import darknet_pan_backbone, convolutional
from .transform import Transformer


class YOLOv5(nn.Module):
    def __init__(self, num_classes, model_size=(0.33, 0.5),
                 match_thresh=4, giou_ratio=1, eps=0, img_sizes=(320, 416),
                 score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        # original
        anchors1 = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        # [320, 416]
        anchors = [
            [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
            [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
            [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
        ]
        # [544, 544]
        anchors3 = [
            [[6.0, 7.6], [12.0, 16.0], [22.3, 26.9]],
            [[31.0, 54.7], [69.4, 38.9], [53.3, 87.5]],
            [[159.0, 101.8], [95.8, 187.0], [309.8, 281.9]],
        ]
        # [448, 608]
        anchors4 = [
            [[8.6, 11.6], [30.5, 18.6], [16.5, 35.0]],
            [[34.7, 60.8], [77.1, 43.0], [58.4, 96.1]],
            [[173.0, 110.8], [104.6, 204.1], [337.5, 305.7]],
        ]
        loss_weights = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.58}
        
        self.backbone = darknet_pan_backbone(
            depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        
        in_channels_list = self.backbone.out_channels_list
        assert len(anchors) == len(in_channels_list)
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes)
        
        self.head = Head(
            predictor, anchors, match_thresh,
            giou_ratio, eps, loss_weights, 
            score_thresh, nms_thresh, detections)
        
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


class Predictor(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            
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
    
    