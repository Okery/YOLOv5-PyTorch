import os
from PIL import Image

import torch
from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, file_root, ann_file, train=False, transforms=None):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.file_root = file_root
        self.train = train
        self.transforms = transforms
        
        self.coco = COCO(ann_file)
        self.ids = tuple(str(k) for k in self.coco.imgs)
        
        self._classes = {k: v["name"] for k, v in self.coco.cats.items()} # original classes
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats)) # dense classes
        
        # The dataset outputs dense labels, thus the labels' value is of range [0, 79].
        # It's necessary to convert resutls' labels to annotation labels.
        self.ann_labels = {self.classes.index(v): k for k, v in self._classes.items()}
        
        dirname, split = os.path.split(file_root)
        checked_id_file = os.path.join(dirname, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.file_root, img_info["file_name"]))
        return image.convert("RGB") # avoid grey image
    
    @staticmethod
    def convert_to_xyxy(box): # box format: (xmin, ymin, w, h)
        x1, y1, w, h = box.T
        new_box = torch.stack((x1, y1, x1 + w, y1 + h), dim=1)
        #new_box = torch.stack((x1 + w / 2, y1 + h / 2, w, h), dim=1)
        return new_box # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        #masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann["bbox"])
                name = self._classes[ann["category_id"]]
                labels.append(self.classes.index(name))
                #mask = self.coco.annToMask(ann)
                #mask = torch.tensor(mask, dtype=torch.uint8)
                #masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            #masks = torch.stack(masks)
        else:
            boxes = torch.empty((0, 4))
            labels = torch.empty((0,), dtype=torch.long)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels)
        return target
    
    