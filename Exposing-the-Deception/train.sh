#!/usr/bin/bash

python training.py  --name FFPP2503152220                                      \
                    --gpu_num 0,1                                                   \
                    --model resnet                    \
                    --epoch 20                                                      \
                    --weight_decay 1e-6                                             \
                    --lr 1e-3                                                       \
                    --bs 256                                                        \
                    --test_bs 1000                                                  \
                    --num_workers 12                                                \
                    --size 224                                                      \
                    --dataset FF++_c23    \
                    --mixup True                                                    \
                    --alpha 0.5                                                     \
                    --lil_loss True                             \
                    --gil_loss True                             \
                    --temperature 1.5                           \
                    --mi_calculator kl                          \
                    --balance_loss_method auto,hyper            \
                    --scales [1,2,10]                           \
                    --num_LIBs 4                                \
                    --test False                                \
                    --save_model True                           \
                    --save_path output                          \