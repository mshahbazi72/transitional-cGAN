#!/bin/bash

tar -xvf datasets/ImageNet_Carnivores_20_100.tar -C ./datasets

python dataset_tool.py --source=datasets/ImageNet_Carnivores_20_100 --dest=datasets/ImageNet_Carnivores_20_100.zip --transform=center-crop --width=128 --height=128



