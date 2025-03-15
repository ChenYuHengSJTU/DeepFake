#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/efn4_fpn_hm_adv.yaml \
                                              -i 447.png

# Current ACC, AUC, AP, AR, mF1 for ['Celeb-real', 'Celeb-synthesis', 'YouTube-real'] --- ['real', 'fake'] --                 100.0 -- nan -- 0.0 -- 100.0 -- 0.0