# YOLOv5-PyTorch
A PyTorch implementation of YOLOv5.

This repository has two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The model is based on [ultralytics' repo](https://github.com/ultralytics/yolov5),
and the code is using the structure of [TorchVision](https://github.com/pytorch/vision).

## Requirements

- **Windows** or **Linux**, with **Python ≥ 3.6**

- **[PyTorch](https://pytorch.org/) ≥ 1.4.0**

- **matplotlib** - visualizing images and results

- **[pycocotools](https://github.com/cocodataset/cocoapi)** - for COCO dataset and evaluation; Windows version is [here](https://github.com/philferriere/cocoapi)

There is a problem with pycocotools for Windows. See [Issue #356](https://github.com/cocodataset/cocoapi/issues/356).

Besides, it's better to remove the prints in pycocotools.

**optional:**

- **nvidia dali (Linux)** - a faster data loader

## Datasets

This repository supports VOC and COCO datasets.

If you want to train your own dataset, you may:

- write the correponding dataset code

- convert your dataset to COCO-style

**PASCAL VOC 2012** ([download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)): ```http://host.robots.ox.ac.uk/pascal/VOC/voc2012/```

**MS COCO 2017**: ```http://cocodataset.org/```

Nvidia DALI is strongly recommended. It's much faster than the original data loader.

Currently this repository supports COCO-style dataset with DALI.

## Training

Train on COCO dataset, using 1 GPU (if you wanna use 2 GPUs, set --nproc_per_node=2):
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --use-cuda --dali --mosaic \
--epochs 190 --data-dir "./data/coco2017" --ckpt-path "yolov5s_coco.pth"
```
A more concrete modification is in ```run.sh```.

To run it:
```
bash ./run.sh
```
If you are using PyTorch ≥ 1.6.0 and RTX series GPUs, the code will enable automatic mixed training (AMP).

## Demo and Evaluation

- Run ```demo.ipynb```.

![example](https://github.com/Okery/YOLOv5-PyTorch/blob/master/images/r000.jpg)

- Modify the parameters in ```eval.ipynb``` to test the model.

## Performance

The model is trained from scratch, on COCO 2017 train, using a single RTX 2080Ti GPU.

One entire epoch takes about 435 seconds (train 405s + eval 30s).

Test on COCO 2017 val:

| model | imgs/s (train) | imgs/s (test) | params | bbox AP | weights | pwd |
| :----: | :---: | :---: | :--: | :--: | :--: | :--: |
| YOLOv5s | 303 | 495 | 7.5M | 36.1 | [yolov5s_768d3ec1](https://pan.baidu.com/s/1eiwY46mpkdEdG_spzoxhpg) | ya7y |
