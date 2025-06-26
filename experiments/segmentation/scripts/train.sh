#!/usr/bin/env bash

# #train
# CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.train --dataset custom \
#     --model deeplab --jpu JPU  --aux --aux-weight 0.4 \
#     --backbone resnet50 --checkname deeplab_res50_custom \
#        #--fusion-net

# test [single-scale]
CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test --dataset custom \
    --model deeplab --jpu JPU --aux --backbone resnet50 --resume experiments/segmentation/runs/custom/deeplab/deeplab_res50_custom/checkpoint.pth  \
    --split test --mode val

# # test [multi-scale]
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset custom \
#     --model deelab --backbone resnet50 --resume {MODEL} \
#     --split val --mode testval --ms

# # predict [single-scale]
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset custom \
#     --model deeplab --backbone resnet50 --resume {MODEL} \

#     --split val --mode test

# # predict [multi-scale]
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset custom \
#     --model deeplab --backbone resnet50 --resume {MODEL} \
#     --split val --mode test --ms

# # fps
# CUDA_VISIBLE_DEVICES=0 python -m experiments.segmentation.test_fps_params --dataset custom \
#     --model deeplab --backbone resnet50
