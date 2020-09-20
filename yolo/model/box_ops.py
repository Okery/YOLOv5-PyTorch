import math

import torch

    
def size_matched_idx(wh1, wh2, thresh):
    #area1 = wh1.prod(1)
    #area2 = wh2.prod(1)
    
    #wh = torch.min(wh1[:, None], wh2[None])
    #inter = wh.prod(2)
    #iou = inter / (area1[:, None] + area2 - inter)
    #return torch.where(iou > thresh)
    
    ratios = wh1[:, None] / wh2[None]
    max_ratios = torch.max(ratios, 1. / ratios).max(2)[0]
    return torch.where(max_ratios < thresh)


def assign_targets_to_proposals(xy, size, overlap=0.5):
    x, y = xy.T
    ids = [torch.arange(len(xy), device=xy.device)]
    
    ids.append(torch.where((x > 1) & (x % 1 < overlap))[0]) # lt_x
    ids.append(torch.where((y > 1) & (y % 1 < overlap))[0]) # lt_y
    ids.append(torch.where((x < size[1] - 1) & (x % 1 > (1 - overlap)))[0]) # rb_x
    ids.append(torch.where((y < size[0] - 1) & (y % 1 > (1 - overlap)))[0]) # rb_y
    
    offsets = xy.new_tensor([[0, 0], [-overlap, 0], [0, -overlap], [overlap, 0], [0, overlap]])
    coordinates = torch.cat([xy[ids[i]] + offsets[i] for i in range(5)]).long()
    return torch.cat(ids), coordinates


# temporarily not merge box_giou and box_ciou
def box_giou(box1, box2): # box format: (cx, cy, w, h)
    cx1, cy1, w1, h1 = box1.T
    cx2, cy2, w2, h2 = box2.T
    
    b1_x1, b1_x2 = cx1 - w1 / 2, cx1 + w1 / 2
    b1_y1, b1_y2 = cy1 - h1 / 2, cy1 + h1 / 2
    b2_x1, b2_x2 = cx2 - w2 / 2, cx2 + w2 / 2
    b2_y1, b2_y2 = cy2 - h2 / 2, cy2 + h2 / 2
    
    ws = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    hs = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = ws.clamp(min=0) * hs.clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union
    
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch
    return iou - (c_area - union) / c_area


def box_ciou(box1, box2): # box format: (cx, cy, w, h)
    cx1, cy1, w1, h1 = box1.T
    cx2, cy2, w2, h2 = box2.T
    
    b1_x1, b1_x2 = cx1 - w1 / 2, cx1 + w1 / 2
    b1_y1, b1_y2 = cy1 - h1 / 2, cy1 + h1 / 2
    b2_x1, b2_x2 = cx2 - w2 / 2, cx2 + w2 / 2
    b2_y1, b2_y2 = cy2 - h2 / 2, cy2 + h2 / 2
   
    ws = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    hs = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = ws.clamp(min=0) * hs.clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union
    
    v = (2 / math.pi * (torch.atan(w2 / h2) - torch.atan(w1 / h1))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v)
        
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2
    return iou - (rho2 / c2 + v * alpha)


def box_iou(box1, box2): # box format: (x1, y1, x2, y2)
    area1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
    area2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)
    
    
def nms(boxes, scores, threshold):
    return torch.ops.torchvision.nms(boxes, scores, threshold)


def batched_nms(boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = labels.to(boxes) * max_size
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, threshold)
    return keep
    
    
def all_batched_nms(ids, boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = torch.stack((labels, ids, labels, ids), dim=1) * max_size
    boxes_for_nms = boxes + offsets
    keep = nms(boxes_for_nms, scores, threshold)
    return keep


def cxcywh2xyxy(box): # box format: (cx, cy, w, h)
    cx, cy, w, h = box.T
    ws = w / 2
    hs = h / 2
    new_box = torch.stack((cx - ws, cy - hs, cx + ws, cy + hs), dim=1)
    return new_box


def xyxy2cxcywh(box): # box format: (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.T
    new_box = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
    return new_box

