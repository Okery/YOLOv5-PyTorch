import os

import torch
from nvidia import dali
from nvidia.dali.plugin.pytorch import feed_ndarray

from .coco_dataset import COCODataset

# Nvidia DALI docs: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

# How to install DALI for CUDA 10.x? 
# 1) download the wheel file: https://developer.download.nvidia.cn/compute/redist/nvidia-dali-cuda100/
# 2) open the terminal, enter: pip install nvidia_dali_cuda100-1.3.0-2471498-py3-none-manylinux2014_x86_64.whl

class COCOPipeline:
    def __init__(self, file_root, ann_file, batch_size, shuffle, num_threads, device_id, world_size):
        super().__init__()
        self.pipe = dali.pipeline.Pipeline(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        with self.pipe:
            jpgs, boxes, labels, img_ids = dali.fn.readers.coco(
                file_root=file_root, annotations_file=ann_file, shuffle_after_epoch=shuffle,
                num_shards=world_size, shard_id=device_id, skip_empty=True, image_ids=True,
                pad_last_batch=True, ltrb=True, read_ahead=True
            )
            images = dali.fn.decoders.image(jpgs, device="mixed")
            self.pipe.set_outputs(images, boxes, labels, img_ids)
        self.pipe.build()

    def run(self):
        images, boxes, labels, img_ids = self.pipe.run()
        return images, boxes, labels, img_ids

class DALICOCODataLoader:
    def __init__(self, file_root, ann_file, batch_size=1, drop_last=False, collate_fn=None, 
                 shuffle=False, num_threads=4, device_id=0, world_size=1):
        self.dataset = COCODataset(file_root, ann_file, train=True)
        shard, extra = len(self.dataset) // world_size, len(self.dataset) % world_size
        self.init_lengths = [shard if i < world_size - extra else shard + 1 for i in range(world_size)]
        self.length = self.init_lengths[device_id]
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.device_id = device_id
        self.world_size = world_size
        self.collate_fn = collate_fn if collate_fn else lambda x: x
        
        self.pipe = COCOPipeline(file_root, ann_file, batch_size, shuffle, num_threads, device_id, world_size)
        
        self.epoch = 0
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        for i in range(len(self)):
            batch = self.preprocess(self.pipe.run())
            if i == len(self) - 1:
                extra = self.length % self.batch_size
                if extra:
                    if self.drop_last:
                        self.pipe.run()
                    else:
                        batch = batch[:extra]
                        
                if not self.shuffle:
                    self.epoch += 1
                    self.length = self.init_lengths[(self.epoch + self.device_id) % self.world_size]
            yield self.collate_fn(batch)
            
    def __len__(self):
        if self.drop_last:
            return self.length // self.batch_size
        return (self.length + self.batch_size - 1) // self.batch_size
    
    def preprocess(self, pipe_out):
        image_list, boxes_list, labels_list, id_list = pipe_out
        
        batch = []
        for i in range(len(image_list)):
            img = torch.empty(image_list[i].shape(), device="cuda", dtype=torch.uint8)
            feed_ndarray(image_list[i], img, self.stream)
            self.stream.synchronize()
            img = img.permute(2, 0, 1) / 255.

            img_id = torch.from_numpy(id_list.at(i)).cuda()
            boxes = torch.from_numpy(boxes_list.at(i)).cuda()
            labels = torch.from_numpy(labels_list.at(i) - 1).cuda()
            tgt = {"image_id": img_id, "boxes": boxes, "labels": labels.long()}
            
            batch.append((img, tgt))
        return batch
    
    