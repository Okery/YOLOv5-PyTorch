import torch
import torch.nn.functional as F
from torch import nn

from . import box_ops


class Head(nn.Module):
    def __init__(self, predictor, anchors, strides, 
                 match_thresh, giou_ratio, loss_weights, 
                 score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.strides = strides
        
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.loss_weights = loss_weights
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections
        
        self.merge = False
        self.eval_with_loss = False
        #self.min_size = 2
        
    def forward(self, features, targets, image_shapes=None, scale_factors=None, max_size=None):
        preds = self.predictor(features)
        
        if self.training:
            losses = self.compute_loss(preds, targets)
            return losses
        else:
            losses = {}
            if self.eval_with_loss:
                losses = self.compute_loss(preds, targets)
                
            results = self.inference(preds, image_shapes, scale_factors, max_size)
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
                
                #pos = 1 - 0.5 * self.eps
                #neg = 1 - pos
                gt_label = torch.zeros_like(pred_level[..., 5:])
                gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1
                losses["loss_cls"] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)
            losses["loss_obj"] += F.binary_cross_entropy_with_logits(pred[..., 4], gt_object)

        losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
        return losses
    
    def inference(self, preds, image_shapes, scale_factors, max_size):
        ids, ps, boxes = [], [], []
        for pred, stride, wh in zip(preds, self.strides, self.anchors): # 3.54s
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            p = pred[n, y, x, a]
            
            xy = torch.stack((x, y), dim=1)
            xy = (2 * p[:, :2] - 0.5 + xy) * stride
            wh = 4 * p[:, 2:4] ** 2 * wh[a]
            box = torch.cat((xy, wh), dim=1)
            
            ids.append(n)
            ps.append(p)
            boxes.append(box)
            
        ids = torch.cat(ids)
        ps = torch.cat(ps)
        boxes = torch.cat(boxes)
        
        boxes = box_ops.cxcywh2xyxy(boxes)
        logits = ps[:, [4]] * ps[:, 5:]
        indices, labels = torch.where(logits > self.score_thresh) # 4.94s
        ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]
        
        results = []
        for i, im_s in enumerate(image_shapes): # 20.97s
            keep = torch.where(ids == i)[0] # 3.11s
            box, label, score = boxes[keep], labels[keep], scores[keep]
            #ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
            #keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
            #boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s
            
            if len(box) > 0:
                box[:, 0].clamp_(0, im_s[1]) # 0.39s
                box[:, 1].clamp_(0, im_s[0]) #~
                box[:, 2].clamp_(0, im_s[1]) #~
                box[:, 3].clamp_(0, im_s[0]) #~
                
                keep = box_ops.batched_nms(box, score, label, self.nms_thresh, max_size) # 4.43s
                keep = keep[:self.detections]
                
                nms_box, nms_label = box[keep], label[keep]
                if self.merge: # slightly increase AP, decrease speed ~14%
                    mask = nms_label[:, None] == label[None]
                    iou = (box_ops.box_iou(nms_box, box) * mask) > self.nms_thresh # 1.84s
                    weights = iou * score[None] # 0.14s
                    nms_box = torch.mm(weights, box) / weights.sum(1, keepdim=True) # 0.55s
                    
                box, label, score = nms_box / scale_factors[i], nms_label, score[keep] # 0.30s
            results.append(dict(boxes=box, labels=label, scores=score)) # boxes format: (xmin, ymin, xmax, ymax)
            
        return results
    