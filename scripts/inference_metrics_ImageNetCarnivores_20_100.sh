#!/bin/bash

python calc_metrics.py \
--network=outputs/00000-ImageNet_Carnivores_20_100-cond-mirror-auto4/network-snapshot.pkl \
--data=datasets/ImageNet_Carnivores_20_100.zip \
--metrics=fid50k_full,kid50k_full,pr50k3_full \
--mirror=1 --gpus=4
