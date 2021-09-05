## YOLOv5-PyTorch

![First](https://github.com/Okery/YOLOv5-PyTorch/blob/master/images/r000.jpg)

A PyTorch implementation of YOLOv5.

This repository has two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The model is based on [ultralytics' repo](https://github.com/ultralytics/yolov5),
and the code is using the structure of [TorchVision](https://github.com/pytorch/vision).

## Requirements

- **Windows** or **Linux**, with **Python ≥ 3.6**

- **[PyTorch](https://pytorch.org/) ≥ 1.6.0**

- **matplotlib** - visualizing images and results

- **[pycocotools](https://github.com/cocodataset/cocoapi)** - for COCO dataset and evaluation; Windows version is [here](https://github.com/philferriere/cocoapi)

There is a problem with pycocotools for Windows. See [Issue #356](https://github.com/cocodataset/cocoapi/issues/356).

Besides, it's better to remove the prints in pycocotools.

**optional:**

- **nvidia dali (Linux only)** - a faster data loader(recommended). [Download page](https://developer.download.nvidia.cn/compute/redist/nvidia-dali-cuda100/)

## Datasets

This repository supports VOC and COCO datasets.

If you want to train your own dataset, you may:

- write the correponding dataset code

- convert your dataset to COCO-style

**PASCAL VOC 2012**: ```http://host.robots.ox.ac.uk/pascal/VOC/voc2012/```

**MS COCO 2017**: ```http://cocodataset.org/```

NVIDIA DALI is strongly recommended. It may be much faster than the original data loader.

Currently this repository supports COCO-style dataset with DALI.

## Model

The model is mainly made of Darknet and PANet.

You can get its flowchart by opening ```images/YOLOv5.drawio``` with [drawio](https://app.diagrams.net/).

## Training

Train on COCO dataset, using 1 GPU (if you wanna use N GPUs, just set --nproc_per_node=N):
```
python -m torch.distributed.run --nproc_per_node=1 --use_env train.py --use-cuda --dali --mosaic \
--epochs 190 --data-dir "./data/coco2017" --ckpt-path "yolov5s_coco.pth"
```
A more concrete modification is in ```run.sh```.

To run it:
```
bash ./run.sh
```
If you use RTX series GPUs, the code will enable automatic mixed training (AMP).

## Demo and Evaluation

- Run ```demo.ipynb```.

![example](https://github.com/Okery/YOLOv5-PyTorch/blob/master/images/r002.jpg)

- Modify the parameters in ```eval.ipynb``` to test the model.

## Performance

Test on COCO 2017 val set, on 1 2080Ti GPU:

The weights is from [ultralytics' repo](https://github.com/ultralytics/yolov5).

| model | bbox AP | FPS | params |
| :----: |:---: | :--: | :--: |
| [YOLOv5s](https://github.com/Okery/YOLOv5-PyTorch/releases/download/v0.3/yolov5s_official_2cf45318.pth) | 36.1 | 410 | 7.5M |
