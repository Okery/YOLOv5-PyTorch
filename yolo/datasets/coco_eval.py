import copy
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        self.has_results = False
            
    def accumulate(self, coco_results):
        if len(coco_results) == 0:
            return
            
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_dt = self.coco_gt.loadRes(coco_results) if coco_results else COCO() # use the method loadRes

            coco_eval.cocoDt = coco_dt 
            coco_eval.params.imgIds = image_ids # ids of images to be evaluated
            coco_eval.evaluate() # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate() # 3s
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print("IoU metric: {}".format(iou_type))
                self.coco_eval[iou_type].summarize()
        else:
            print("evaluation has no results")

            
def prepare_for_coco(predictions, ann_labels):
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        # convert to coco bbox format: xmin, ymin, w, h
        boxes = prediction["boxes"]
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)

        boxes = boxes.tolist()

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        labels = [ann_labels[l] for l in labels]

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

