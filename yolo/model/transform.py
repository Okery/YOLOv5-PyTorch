import copy
import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        #self.max_size = math.ceil(max_size / stride) * stride
        #self.grid_size = range(math.ceil(min_size / stride), math.ceil(max_size / stride) + 1)
        
        self.mean = image_mean
        self.std = image_std
        self.stride = stride
        
        #self.accumulate = 10
        #self.count = 0
        self.flip_prob = 0.5
        self.mosaic = False
        
    def forward(self, images, targets):
        images = [img for img in images]
        targets = copy.deepcopy(targets) if targets is not None else None 
        
        #if self.training:
        #    if self.count % self.accumulate == 0:
        #        self.max_size = random.choice(self.grid_size) * self.stride
        #else:
        #    self.max_size = self.grid_size[-1] * self.stride
            
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None
            
            image, target = self.transform(image, target)
            
            images[i] = image
            if target is not None:
                targets[i] = target
                
        if self.training and self.mosaic:
            images, targets = mosaic_augment(images, targets)
            
        image_shapes = [img.shape[1:] for img in images]
        images = self.batched_images(images)
        #self.count += 1
        return images, targets, image_shapes

    def normalize(self, image):
        mean = image.new(self.mean)[:, None, None]
        std = image.new(self.std)[:, None, None]
        return (image - mean) / std
    
    def transform(self, image, target):
        if self.training:
            if random.random() < self.flip_prob:
                image, target["boxes"] = self.horizontal_flip(image, target["boxes"])
        
        image, target = self.resize(image, target)
        image = self.normalize(image)
        return image, target
        
    def horizontal_flip(self, image, boxes):
        w = image.shape[2]
        image = image.flip(2)
        
        tmp = boxes[:, 0] + 0
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - tmp
        return image, boxes

    def resize(self, image, target):
        orig_image_shape = image.shape[-2:]
        min_size = min(orig_image_shape)
        max_size = max(orig_image_shape)
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        #scale_factor = self.max_size / max(orig_image_shape)
        if scale_factor != 1:
            size = [round(s * scale_factor) for s in orig_image_shape]
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]

            if target is not None:
                box = target["boxes"]
                box[:, [0, 2]] *= size[1] / orig_image_shape[1]
                box[:, [1, 3]] *= size[0] / orig_image_shape[0]
        return image, target
    
    def batched_images(self, images):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / self.stride) * self.stride for m in max_size)

        batch_shape = (len(images), 3,) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs
    
    def postprocess(self, results, image_shapes, orig_image_shapes):
        for i, (res, im_s, o_im_s) in enumerate(zip(results, image_shapes, orig_image_shapes)):
            boxes = res["boxes"]
            boxes[:, [0, 2]] *= o_im_s[1] / im_s[1]
            boxes[:, [1, 3]] *= o_im_s[0] / im_s[0]
            
        return results
    
    
def sort_images1(shapes, out):
    # deprecated
    shapes.sort(key=lambda x: x[0]) # max h
    img1 = shapes.pop()
    img2 = shapes.pop()
    out.append(img1[2])
    out.append(img2[2])
    
    shapes.sort(key=lambda x: abs(x[1] - img1[1]), reverse=True) # similar w
    img3 = shapes.pop()
    out.append(img3[2])
    
    shapes.sort(key=lambda x: abs(x[0] - img3[0]) + abs(x[1] - img2[1]), reverse=True) # similar h, w
    img4 = shapes.pop()
    out.append(img4[2])
    if shapes:
        sort_images(shapes, out)
        
        
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
    
