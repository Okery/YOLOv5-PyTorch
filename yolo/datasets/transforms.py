import math
import random
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["RandomAffine"]


class RandomAffine:
    def __init__(self, degrees=(0, 0), translate=(0, 0),
                 scale=(1, 1), shear=(0, 0, 0, 0), **kwargs):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.kwargs = kwargs

    @staticmethod
    def get_params(img_size, degrees, translate, scale_ranges, shears):
        angle = random.uniform(degrees[0], degrees[1])
        ms = [s * t for s, t in zip(img_size, translate)]
        translations = [round(random.uniform(-m, m)) for m in ms]
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
        shear = [random.uniform(shears[0], shears[1]),
                 random.uniform(shears[2], shears[3])]
        
        rot = math.radians(angle)
        shear = [math.radians(s) for s in shear]
        return rot, translations, scale, shear
    
    def __call__(self, image, target={}):
        ret = self.get_params(image.size, self.degrees, self.translate, self.scale, self.shear)
        return self.affine(image, target, *ret, **self.kwargs)

    @staticmethod
    def affine(image, target, rot=0, translate=(0, 0), scale=1, shear=(0, 0), **kwargs):
        """
        Arguments:
            image (PIL.Image)
            target (dict[tensor])
            rot (float): radian, 0~3.14
            translate (tuple[float, float]): [0.0, 1.0]
            scale (float): > 0.0
            shear (tuple[float, float]): radian, [-3.14, 3.14]
        """
        w, h = image.size
        cx, cy = w * 0.5 + 0.5, h * 0.5 + 0.5
        tx, ty = translate
        sx, sy = shear

        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)
        M[2] += cx
        M[5] += cy

        if target:
            boxes = target["boxes"]
            labels = target["labels"]

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            nxs = (boxes[:, [0, 2, 2, 0]] - cx) * scale
            nys = (boxes[:, [1, 1, 3, 3]] - cy) * scale
            xs = nxs * a + nys * b + cx + tx
            ys = nxs * c + nys * d + cy + ty

            xy = torch.stack((xs, ys), dim=2)
            fx1, fy1, fx2, fy2 = torch.cat((xy.min(1)[0], xy.max(1)[0]), dim=1).T

            fx1.clamp_(0, w), fx2.clamp_(0, w)
            fy1.clamp_(0, h), fy2.clamp_(0, h)

            new_boxes = torch.stack((fx1, fy1, fx2, fy2), dim=1)
            ws, hs = fx2 - fx1, fy2 - fy1
            new_areas = ws * hs
            keep = torch.where((ws > 2) & (hs > 2) & ((new_areas / areas) > 0.2))[0]

            target["boxes"] = new_boxes[keep]
            target["labels"] = labels[keep]

        """
        # tensor version of affine transform, but it is slower than PIL.Image version
        y, x =torch.meshgrid(torch.arange(h), torch.arange(w))
        nx = 2 * (M[0] * x + M[1] * y + M[2]) / w - 1
        ny = 2 * (M[3] * x + M[4] * y + M[5]) / h - 1
        grid = torch.stack((nx, ny), dim=2)[None].to(image.device)
        image = F.grid_sample(image[None], grid, align_corners=False)[0]
        """
        image = image.transform((w, h), Image.AFFINE, M, **kwargs)
        return image, target

    