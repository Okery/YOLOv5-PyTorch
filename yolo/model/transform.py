import copy
import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        
        self.flip_prob = 0.5
        self.mosaic = False
        
    def forward(self, images, targets):
        if targets is None:
            transformed = [self.transforms(img, targets) for img in images]
        else:
            targets = copy.deepcopy(targets)
            transformed = [self.transforms(img, tgt) for img, tgt in zip(images, targets)]
        
        images, targets, scale_factors = zip(*transformed)
            
        image_shapes = None
        if self.training:
            if self.mosaic:
                images, targets = mosaic_augment(images, targets)
        else:
            image_shapes = [img.shape[1:] for img in images]
            
        images = self.batch_images(images)
        return images, targets, scale_factors, image_shapes

    #def normalize(self, image):
    #    mean = image.new(self.mean)[:, None, None]
    #    std = image.new(self.std)[:, None, None]
    #    return (image - mean) / std
    
    def transforms(self, image, target):
        image, target, scale_factor = self.resize(image, target)
        
        if self.training:
            if random.random() < self.flip_prob:
                image, target["boxes"] = self.horizontal_flip(image, target["boxes"])
            
        return image, target, scale_factor
        
    def horizontal_flip(self, image, boxes):
        w = image.shape[2]
        image = image.flip(2)
        
        tmp = boxes[:, 0] + 0
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - tmp
        return image, boxes

    def resize(self, image, target):
        orig_image_shape = image.shape[1:]
        min_size = min(orig_image_shape)
        max_size = max(orig_image_shape)
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        
        if scale_factor != 1:
            size = [round(s * scale_factor) for s in orig_image_shape]
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]

            if target is not None:
                box = target["boxes"]
                box[:, [0, 2]] *= size[1] / orig_image_shape[1]
                box[:, [1, 3]] *= size[0] / orig_image_shape[0]
        return image, target, scale_factor
    
    def batch_images(self, images):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / self.stride) * self.stride for m in max_size)

        batch_shape = (len(images), 3,) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs
        
        
def sort_images(shapes, out, dim):
    shapes.sort(key=lambda x: x[dim])
    out.append(shapes.pop()[2])
    if dim == 0:
        out.append(shapes.pop()[2])
        out.append(shapes.pop(1)[2])
    else:
        out.append(shapes.pop(1)[2])
        out.append(shapes.pop()[2])
    out.append(shapes.pop(0)[2])
    if shapes:
        sort_images(shapes, out, dim)
        
        
def mosaic_augment(images, targets):
    assert len(images) % 4 == 0, "mosaic augmentation: len(images) % 4 != 0"
    shapes = [(img.shape[1], img.shape[2], i) for i, img in enumerate(images)]
    ratios = [int(h >= w) for h, w, _ in shapes]
    dim = int(sum(ratios) >= len(ratios) * 0.5)
    order = []
    sort_images(shapes, order, dim)
    
    new_images, new_targets = [], []
    for i in range(len(order) // 4):
        hs, ws = zip(*[images[o].shape[-2:] for o in order[4 * i:4 * (i + 1)]])
        tl_y, br_y = max(hs[0], hs[1]), max(hs[2], hs[3])
        tl_x, br_x = max(ws[0], ws[2]), max(ws[1], ws[3])
        merged_image = images[0].new_zeros((3, tl_y + br_y, tl_x + br_x))

        for j in range(4):
            index = order[4 * i + j]
            img = images[index]
            box = targets[index]["boxes"]
            h, w = img.shape[-2:]

            x1, y1, x2, y2 = tl_x, tl_y, tl_x, tl_y
            if j == 0: # top left
                x1 -= w
                y1 -= h
            elif j == 1: # top right
                x2 += w
                y1 -= h
            elif j == 2: # bottom left
                x1 -= w
                y2 += h
            elif j == 3: # bottom right
                x2 += w
                y2 += h

            merged_image[:, y1:y2, x1:x2].copy_(img)
            box[:, [0, 2]] += x1
            box[:, [1, 3]] += y1

        boxes = torch.cat([targets[o]["boxes"] for o in order[4 * i:4 * (i + 1)]])
        labels = torch.cat([targets[o]["labels"] for o in order[4 * i:4 * (i + 1)]])
        
        new_images.append(merged_image)
        new_targets.append(dict(boxes=boxes, labels=labels))
    return new_images, new_targets
    
