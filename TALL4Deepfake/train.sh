#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset ffpp \
 --input-size 112 --num_clips 8 --output_dir ./log --opt adamw --lr 1.5e-5 --warmup-lr 1.5e-8 --min-lr 1.5e-7 \
 --epochs 60 --sched cosine --duration 4 --batch-size 4 --thumbnail_rows 2 --disable_scaleup --cutout True \
 --warmup-epochs 10 --no-amp --model TALL_SWIN \
 --hpe_to_token 2>&1 | tee ./output/train_ffpp_`date +'%m_%d-%H_%M'`.log