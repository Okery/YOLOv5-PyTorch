"""

The code of this file is only a copy of Jupyter Notebook code snippet.
It's a MESS

"""
# -----------------------------------------------------------------------------------------------------
import torch
from collections import OrderedDict

a = m2.state_dict()
b = c2
n = []
for k1, k2 in zip(a, b):
    n.append((k1, b[k2]))
torch.save(OrderedDict(n), "xxx.pth")
# -----------------------------------------------------------------------------------------------------
import torch
import yolo

def _range(*args):
    return list(range(*args))

def bottle2concat(s, l):
    tmp = [range(s,s+6),range(s+19,s+19+l*12),range(s+7,s+5,-1),range(s+14,s+19),range(s+8,s+14)]
    order = [o for r in tmp for o in r]
    return order

ckpt = torch.load("/data/nextcloud/dbc2017/files/jupyter/data/ckpt/converted_yolov5s.pt")
msd = ckpt["model"]
del msd["model.27.anchors"], msd["model.27.anchor_grid"]

model = yolo.YOLOv5(80)
tgt = model.state_dict()

stages = [12, 49, 110]
layers = [1, 3, 3]
orders = []
for s, l in zip(stages, layers):
    cur = _range(s - 6, s) + bottle2concat(s, l)
    orders.extend(cur)
    
backbone = _range(0, 6) + orders + _range(165, 183)


stages = [183, 220, 257, 296, 335]
layers = [1, 1, 1, 1, 1]
csp = [bottle2concat(s, l) for s, l in zip(stages, layers)]
conv = [_range(x, x + 6) for x in [214, 251, 290, 329]]
pred = [_range(x, x + 2) for x in [288, 327, 366]]

new = [csp[2],csp[1],csp[0],conv[1],conv[0],conv[2],conv[3],csp[3],csp[4],pred[0],pred[1],pred[2]]
panet = []
for n in new:
    panet.extend(n)
    
total = backbone + panet
#print(len(panet), len(set(panet)), max(panet) - min(panet) + 1)
#print(panet)
norm = {total.index(i): v for i, v in enumerate(msd.values())}
total_new = {k: norm[i] for i, k in enumerate(tgt.keys())}

model.load_state_dict(total_new)
torch.save({"ema": (total_new, 0)}, "/data/nextcloud/dbc2017/files/jupyter/data/ckpt/total_new.pth")


# -----------------------------------------------------------------------------------------------------
import torch
import math
import yolo


torch.set_printoptions(precision=1)

def parse_dataset(ds, img_sizes):
    info = {}
    ids = [int(id_) for id_ in ds.ids]
    
    min_size, max_size = img_sizes
    factor = lambda s: min(min_size / min(s), max_size / max(s))
    scales, pixels = [], []
    for id_ in ids: 
        v = ds.coco.imgs[id_]
        h, w = v["height"], v["width"]
        scale = factor((h, w))
        scales.append(scale)
        pixels.append(round(scale * h) * round(scale * w))

    boxes = torch.cat([ds.get_target(id_)["boxes"] * scale
                       for id_, scale in zip(ds.ids, scales)])
    
    info["boxes"] = boxes
    info["scales"] = scales
    info["pixels"] = pixels
    return info


def wh_iou(wh1, wh2):
    area1 = torch.prod(wh1, dim=1)
    area2 = torch.prod(wh2, dim=1)
    wh = torch.min(wh1[:, None], wh2[None])
    inter = torch.prod(wh, dim=2)
    return inter / (area1[:, None] + area2 - inter)


def kmeans(boxes, k, n=100):
    num = len(boxes)
    last_clusters = torch.zeros((num,))
    rng = torch.randperm(num)[:k]
    clusters = boxes[rng]

    for _ in range(n):
        #iou = wh_iou(boxes, clusters)
        #avg_iou = torch.max(iou, dim=1)[0].mean().item()
        #print("{:.1f} ".format(100 * avg_iou), end="")
        
        nearest_clusters = matched_fn(boxes, clusters)[1]

        if (last_clusters == nearest_clusters).all():
            #print("nice ", end="")
            break

        for i in range(k):
            order = nearest_clusters == i
            clusters[i] = torch.median(boxes[order], dim=0)[0]
        last_clusters = nearest_clusters
            
    iou = wh_iou(boxes, clusters)
    avg_iou = torch.max(iou, dim=1)[0].mean().item()
    return clusters, avg_iou


def matched_fn(wh1, wh2):
    ratios = wh1[:, None] / wh2[None]
    max_ratios = torch.max(ratios, 1 / ratios).max(2)[0]
    return max_ratios.min(1)


def lazy_fn(boxes, k=3, n=5, iters=200, thresh=4):
    results = []
    for _ in range(n):
        clusters, avg_iou = kmeans(boxes, k, iters)
        area = clusters.prod(1)
        order = area.sort()[1]
        clusters = clusters[order]
        
        values = matched_fn(boxes, clusters)[0]
        bpr = (values < thresh).float().mean()
        
        left = len(boxes) * (1 - bpr)
        results.append((clusters, round(100 * avg_iou, 2), round(100 * bpr.item(), 2)))
    return results


def auto_anchors(ds, img_sizes=[320, 416], levels=3, ks=3, stride=4, **kwargs):
    if isinstance(img_sizes, int):
        img_sizes = [img_sizes, img_sizes]
    if isinstance(ks, int):
        ks = [ks] * levels
    assert len(ks) == levels, "len(ks) != levels"
    print(img_sizes, stride)
    
    info = parse_dataset(ds, img_sizes)
    boxes = info["boxes"]
    wh = torch.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1)
    
    new_boxes = wh[(wh >= 2).all(1)]
    print(f"len(new_boxes)={len(new_boxes)}")
    areas = new_boxes.prod(1)

    chunked_areas = areas.sort()[0].chunk(levels)
    print("before split:", [len(x) for x in chunked_areas])
    sep = lambda x: math.ceil(x.sqrt().item() / stride) * stride
    sizes = [0] + [sep(chunked_areas[i][-1]) for i in range(levels - 1)] + [10000]
    #sizes = [0, 72, 180, 10000]
    print("split size:", sizes)
    
    orders = [(areas >= s ** 2) & (areas < sizes[i + 1] ** 2)
              for i, s in enumerate(sizes[:-1])]
    print("after split:", [x.sum().item() for x in orders])

    anchors = []
    for i in range(levels):
        boxes_part = new_boxes[orders[i]]
    
        res = lazy_fn(boxes_part, ks[i], **kwargs)
        res.sort(key=lambda x: x[2])
        print(f"\nlevel {i + 1}:")
        print(*res[-1])
        anchors.append(res[-1][0].tolist())
        
    print("\ntotal: [")
    for i, anchor in enumerate(anchors):
        print("    [", end="")
        for j, anc in enumerate(anchor):
            print([round(a, 1) for a in anc], end=", " if j < len(anchor) - 1 else "")
        print("],")
    print("]")
    
    #values = matched_fn(boxes, clusters)[0]
    #bpr = (values < thresh).float().mean()
    
# original boxes: len=860001
ds = yolo.datasets("coco", "E:/PyTorch/data/coco2017", "train2017", train=True)
auto_anchors(ds, [17*32, 17*32])

"""
[320, 416] 4
len(new_boxes)=855581
before split: [285194, 285194, 285193]
split size: [0, 24, 68, 10000]
after split: [296119, 284202, 275260]

level 1:
tensor([[ 6.1,  8.1],
        [20.6, 12.6],
        [11.2, 23.7]]) 57.77 99.81

level 2:
tensor([[36.2, 26.8],
        [25.9, 57.2],
        [57.8, 47.9]]) 63.5 99.9

level 3:
tensor([[122.1,  78.3],
        [ 73.7, 143.8],
        [236.1, 213.1]]) 60.64 99.96

total: [
    [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
    [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
    [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
]


[544, 544] 4
len(new_boxes)=858492
before split: [286164, 286164, 286164]
split size: [0, 32, 88, 10000]
after split: [301035, 275650, 281807]

level 1:
tensor([[ 6.0,  7.6],
        [12.0, 16.0],
        [22.3, 26.9]]) 56.47 99.5

level 2:
tensor([[31.0, 54.7],
        [69.4, 38.9],
        [53.3, 87.5]]) 64.11 99.9

level 3:
tensor([[159.0, 101.8],
        [ 95.8, 187.0],
        [309.8, 281.9]]) 60.28 99.96

total: [
    [[6.0, 7.6], [12.0, 16.0], [22.3, 26.9]],
    [[31.0, 54.7], [69.4, 38.9], [53.3, 87.5]],
    [[159.0, 101.8], [95.8, 187.0], [309.8, 281.9]],
]
"""
# -----------------------------------------------------------------------------------------------------
import yolo
import torch

model = yolo.YOLOv5(80)

save = [(1, 2)]
fn = lambda x: x if isinstance(x, int) else x[0]
n = 0
for k, v in model.backbone.body.named_modules():
    if isinstance(v, (torch.nn.Conv2d, )):
        if "part2" not in k:
            #pass
            print(n, k, v)
            n += 1
            save.append((fn(v.kernel_size), fn(v.stride)))
            
rf = [1]
ts = 1
for i, (k, s) in enumerate(save):
    r = rf[-1] + (k - 1) * ts
    ts *= s
    rf.append(r)
    print(i - 1, rf[i], ts)
    
# -----------------------------------------------------------------------------------------------------    