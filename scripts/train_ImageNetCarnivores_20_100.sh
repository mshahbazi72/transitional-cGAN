#!/bin/bash
source ~/envs/ada/bin/activate; module load cuda/11.2.2 ninja

python train.py --outdir=/cluster/work/cvl/mshahbazi/stylegan/training-runs \
--data=datasets/ImageNet_Carnivores_20_100.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=4 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full



