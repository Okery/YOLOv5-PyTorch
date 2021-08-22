#!/bin/bash


# directory structure:
# root/
#   input/
#     data/
#       coco2017/
#         train2017/
#         val2017/
#         ...
#     ckpts/
#       mosaic_yolov5s_coco-190.pth
#       results.pth
#     scripts/
#       okery/
#         yolo/
#         run.sh
#         train.py
#         demo.ipynb
#         ...
#   data/
#     logs/
#       log.txt

root="/data/nextcloud/dbc2017/files/jupyter/"
script_dir="okery/"

ngpu=1
dataset="coco"
batch_size=64
print_freq=100
lr=0.01
epochs=10
period=300
img_size1=320
img_size2=416
ckpt_file="yolov5s_${dataset}.pth"
iters=-1

if [ $dataset = "voc" ]
then
    data_dir="${root}input/data/voc2012/VOCdevkit/VOC2012/"
elif [ $dataset = "coco" ]
then
    data_dir="${root}input/data/coco2017/"
fi


setsid python -m torch.distributed.run --nproc_per_node=${ngpu} --use_env ${root}input/scripts/${script_dir}train.py \
--use-cuda --epochs ${epochs} --period ${period} --batch-size ${batch_size} --lr ${lr} --img-sizes ${img_size1} ${img_size2} \
--dataset ${dataset} --data-dir ${data_dir} --iters ${iters} --root ${root} --mosaic --dali \
--ckpt-path ${root}input/ckpts/${ckpt_file} --print-freq ${print_freq} > ${root}data/logs/log.txt 2>&1 &
#
