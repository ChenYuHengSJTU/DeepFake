#!/usr/bin/bash

python package_utils/images_crop.py -d FaceShifter \
-c c23 \
-t test

python package_utils/images_crop.py -d Face2Face \
-c c23 \
-t test

python package_utils/images_crop.py -d NeuralTextures \
-c c23 \
-t test