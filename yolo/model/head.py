import math

import torch
import torch.nn.functional as F
from torch import nn

from . import box_ops


class Head(nn.Module):
    def __init__(self, predictor, anchors, match_thresh,
                 giou_ratio, eps, loss_weights, 
                 score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        
        #self.anchors = []
        #for i, anc in enumerate(anchors, 1):
        #    self.register_buffer("anchor{}".format(i), torch.Tensor(anc))
        #    self.anchors.append(eval("self.anchor{}".format(i)))
        self.register_buffer("anchors", torch.Tensor(anchors))
        
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.eps = eps
        self.loss_weights = loss_weights
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections
        
        self.strides = None
        self.eval_with_loss = False
        #self.min_size = 2
        
    def forward(self, features, batch_shape, targets, image_shapes):
        preds = self.predictor(features)
        
        if self.strides is None:
            self.strides = [2 ** int(math.log2(batch_shape[0] / f.shape[2])) for f in features]

        results, losses = [], {}
        if self.training:
            losses = self.compute_loss(preds, targets)
        else:
            if self.eval_with_loss:
                losses = self.compute_loss(preds, targets)
            results = self.inference(preds, image_shapes, max(batch_shape))
        return results, losses
        
    def compute_loss(self, preds, targets):
        dtype = preds[0].dtype
        image_ids = torch.cat([torch.full_like(tgt["labels"], i)
                                   for i, tgt in enumerate(targets)])
        gt_labels = torch.cat([tgt["labels"] for tgt in targets])
        gt_boxes = torch.cat([tgt["boxes"] for tgt in targets])
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)

        losses = {
            "loss_box": gt_boxes.new_tensor(0),
            "loss_obj": gt_boxes.new_tensor(0),
            "loss_cls": gt_boxes.new_tensor(0)}
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:, 2:], self.match_thresh)

            gt_object = torch.zeros_like(pred[..., 4])
            if len(anchor_id) > 0:
                gt_box_xy = gt_boxes[:, :2][gt_id]
                ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
                anchor_id, gt_id = anchor_id[ids], gt_id[ids]
                image_id = image_ids[gt_id]
                
                pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
                
                xy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
                wh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * wh[anchor_id] / stride
                box_grid = torch.cat((xy, wh), dim=1)
                giou = box_ops.box_giou(box_grid, gt_boxes[gt_id] / stride).to(dtype)
                losses["loss_box"] += (1 - giou).mean()
                
                gt_object[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id] = \
                self.giou_ratio * giou.detach().clamp(0) + (1 - self.giou_ratio)
                
                pos = 1 - 0.5 * self.eps
                neg = 1 - pos
                gt_label = torch.full_like(pred_level[..., 5:], neg)
                gt_label[range(len(gt_id)), gt_labels[gt_id]] = pos
                losses["loss_cls"] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)
            losses["loss_obj"] += F.binary_cross_entropy_with_logits(pred[..., 4], gt_object)

        losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
        return losses
    
    def inference(self, preds, image_shapes, max_size):
        dtype = preds[0].dtype
        
        xs = []
        for pred, stride, wh in zip(preds, self.strides, self.anchors): # 3.54s
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            xy = torch.stack((x, y), dim=1)
            
            xy = (2 * pred[n, y, x, a, :2] - 0.5 + xy) * stride
            wh = 4 * pred[n, y, x, a, 2:4] ** 2 * wh[a]
            pred[n, y, x, a, :4] = torch.cat((xy, wh), dim=1).to(dtype)
            xs.append(pred.flatten(1, 3))
        xs = torch.cat(xs, dim=1)
        
        results = []
        for x, im_s in zip(xs, image_shapes): # 20.97s
            keep = torch.where(x[:, 4] > self.score_thresh)[0] # 3.11s
            x = x[keep] # 0.16s
            boxes, objectness, logits = x[:, :4], x[:, [4]], x[:, 5:] # 1.71s
            boxes = box_ops.cxcywh2xyxy(boxes).float() # 1.58s
            
            boxes[:, 0].clamp_(0, im_s[1]) # 0.39s
            boxes[:, 1].clamp_(0, im_s[0]) #~
            boxes[:, 2].clamp_(0, im_s[1]) #~
            boxes[:, 3].clamp_(0, im_s[0]) #~
            
            #ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
            #keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
            #boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s
            
            logits = logits * objectness # 0.1s
            ids, labels = torch.where(logits > self.score_thresh) # 4.94s
            boxes, scores = boxes[ids], logits[ids, labels]
            
            if len(boxes) > 0:
                keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh, max_size) # 4.43s
                keep = keep[:self.detections]
                
                iou = box_ops.box_iou(boxes[keep], boxes) > self.nms_thresh # 1.84s
                weights = iou * scores[None] # 0.14s
                boxes = torch.mm(weights, boxes) / weights.sum(1, keepdim=True) # 0.55s
                scores, labels = scores[keep], labels[keep] # 0.30s
                
            results.append(dict(boxes=boxes, labels=labels, scores=scores)) # boxes format: (xmin, ymin, xmax, ymax)
        return results
        
    