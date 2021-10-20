import bisect
from collections import defaultdict

import torch

from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

__all__ = ["datasets", "collate_wrapper", "GroupedBatchSampler", "DataPrefetcher"]


def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ["voc", "coco"]
    if ds == choice[0]:
        return VOCDataset(*args, **kwargs)
    if ds == choice[1]:
        return COCODataset(*args, **kwargs)
    else:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))
    
    
def collate_wrapper(batch):
    return CustomBatch(batch)

    
class CustomBatch:
    def __init__(self, data):
        self.images, self.targets = zip(*data)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = [img.pin_memory() for img in self.images]
        self.targets = [{k: v.pin_memory() for k, v in tgt.items()} for tgt in self.targets]
        return self  


class GroupedBatchSampler:
    def __init__(self, sampler, aspect_ratios, batch_size, drop_last=False, factor=3):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        bins = (2 ** torch.linspace(-1, 1, 2 * factor + 1)).tolist()
        self.group_ids = tuple(bisect.bisect(bins, float(x)) for x in aspect_ratios)

    def __iter__(self):
        buffer_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]

        num_remaining = len(self) - num_batches
        
        if num_remaining > 0:
            remaining_ids = []
            for k in sorted(buffer_per_group):
                remaining_ids.extend(buffer_per_group[k])

            for i in range(num_remaining):
                yield remaining_ids[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size # drop last
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    

class DataPrefetcher:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.dataset = data_loader.dataset
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        for i, d in enumerate(self.loader, 1):
            if i == 1:
                d.images = [img.cuda(non_blocking=True) for img in d.images]
                d.targets = [{k: v.cuda(non_blocking=True) for k, v in tgt.items()} for tgt in d.targets]
                self._cache = d
                continue
               
            torch.cuda.current_stream().wait_stream(self.stream)
            out = self._cache
            
            with torch.cuda.stream(self.stream):
                d.images = [img.cuda(non_blocking=True) for img in d.images]
                d.targets = [{k: v.cuda(non_blocking=True) for k, v in tgt.items()} for tgt in d.targets]
            self._cache = d
                
            yield out
        yield self._cache
        
    def __len__(self):
        return len(self.loader)
    
