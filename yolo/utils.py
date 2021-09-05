import copy
import glob
import hashlib
import math
import os
import random
import re
import numpy as np

import torch

__all__ = ["setup_seed", "save_ckpt", "Meter", "ModelEMA", "find_ckpts", 
          "reduce_weights"]


def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    

def save_ckpt(model, optimizer, epochs, path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)

    
def find_ckpts(path):
    prefix, ext = os.path.splitext(path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    return ckpts


def reduce_weights(path):
    ckpt = torch.load(path, torch.device("cpu"))
    if "ema" in ckpt:
        weights = ckpt["ema"][0]
    else:
        weights = ckpt
    for k, v in weights.items():
        if v.is_floating_point():
            weights[k] = v.half()

    sha256 = hashlib.sha256(bytes(str(weights), encoding="utf-8")).hexdigest()

    name, ext = os.path.splitext(path)
    new_file = "{}_{}{}".format(name, sha256[:8], ext)
    torch.save(weights, new_file)
    
    
class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        if len(values) > 0:
            values = [int(v) / 10 for v in values]
            result = {"bbox AP": values[0]} if values else {}
            return result
        else:
            return txt
    
    
class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)
    

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if self.distributed:
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
                   
                