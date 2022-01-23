## Collapse by Conditioning: Training Class-conditional GANs with Limited Data

[Mohamad Shahbazi](https://people.ee.ethz.ch/~mshahbazi/), [Martin Danelljan](https://martin-danelljan.github.io/), [Danda P. Paudel](https://people.ee.ethz.ch/~paudeld/), [Luc Van Gool](https://scholar.google.ch/citations?hl=en&user=TwMib_QAAAAJ)<br>
Paper: https://openreview.net/forum?id=7TZeCsNOUB_<br>

![Teaser image](./docs/main.png)

## Abstract
*Class-conditioning offers a direct means of controlling a Generative Adversarial Network (GAN) based on a discrete input variable. While necessary in many applications, the additional information provided by the class labels could even be expected to benefit the training of the GAN itself. Contrary to this belief, we observe that class-conditioning causes mode collapse in limited data settings, where unconditional learning leads to satisfactory generative ability. Motivated by this observation, we propose a training strategy for conditional GANs (cGANs) that effectively prevents the observed mode-collapse by leveraging unconditional learning. Our training strategy starts with an unconditional GAN and gradually injects conditional information into the generator and the objective function. The proposed method for training cGANs with limited data results not only in stable training but also in generating high-quality images, thanks to the early-stage exploitation of the shared information across classes. We analyze the aforementioned mode collapse problem in comprehensive experiments on four datasets. Our approach demonstrates outstanding results compared with state-of-the-art methods and established baselines.*


## Overview
1. [Requirements](#Requirements)
2. [Getting Started](#Start)
3. [Dataset Prepration](#Data)
4. [Training](#Training)
5. [Evaluation and Logging](#Evaluation)
6. [Contact](#Contact)
8. [How to Cite](#How-to-Cite)


## Requirements<a name="Requirements"></a>

* Linux and Windows are supported, but Linux is recommended for performance and compatibility reasons.
* For the batch size of 64, we have used 4 NVIDIA GeForce RTX 2080 Ti GPUs (each having 11 GiB of memory).
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch installation instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090.  (Why is a separate CUDA toolkit installation required?  See comments of this Github [issue](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/2#issuecomment-779457121).)
* Python libraries: `pip install wandb click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.
* This project uses Weights and Biases for visualization and logging. In addition to installing W&B (included in the command above), you need to [create](https://wandb.ai/login?signup=true) a free account on W&B website. Then, you must login to your account in the command line using the command ‍‍‍`wandb login` (The login information will be asked after running the command).
* Docker users: use the [provided Dockerfile](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/Dockerfile) by StyleGAN2+ADA (./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Getting Started<a name="Start"></a>

The code for this project is based on the [Pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch) of StyleGAN2+ADA. Please first read the [instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md) provided for StyleGAN2+ADA. Here, we mainly provide the additional details required to use our method.

For a quick start, we have provided example scripts in `./scripts`, as well as an example dataset (a tar file containing a subset of ImageNet Carnivores dataset used in the paper) in `./datasets`. Note that the scripts do not include the command for activating python environments. Moreover, the paths for the dataset and output directories can be modified in the scripts based on your own setup.

The following command runs a script that extracts the tar file and creates a ZIP file in the same directory. 
```.bash
bash scripts/prepare_dataset_ImageNetCarnivores_20_100.sh
```
The ZIP file is later used for training and evaluation. For more details on how to use your custom datasets, see [Dataset Prepration](#Data).

Following command runs a script that trains the model using our method with default hyper-parameters:
```.bash
bash scripts/train_ImageNetCarnivores_20_100.sh
```
For more details on how to use your custom datasets, see [Training](#Training)

To calculate the evaluation metrics on a pretrained model, use the following command:
```.bash
bash scripts/inference_metrics_ImageNetCarnivores_20_100.sh
```


Outputs from the training and inferenve commands are by default placed under `out/`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.


## Dataset Prepration<a name="Data"></a>

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. 

Custom datasets can be created from a folder containing images (each sub-directory containing images of one class in case of multi-class datasets) using `dataset_tool.py`; Here is an example of how to convert the dataset folder to the desired ZIP file:
```.bash
python dataset_tool.py --source=datasets/ImageNet_Carnivores_20_100 --dest=datasets/ImageNet_Carnivores_20_100.zip --transform=center-crop --width=128 --height=128
```

The above example reads the images from the image folder provided by `--src`, resizes the images to the sizes provided by `--width` and `--height`, and applys the transform `center-crop` to them. The resulting images along with the metadata (label information) are stored as a ZIP file determined by `--dest`.
see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets) for more details on specific datasets or Legacy TFRecords datasets .

The created ZIP file can be passed to the training and evaluation code using `--data` argument.

## Training<a name="Training"></a>

Training new networks can be done using `train.py`. In order to perform the training using our method, the argument `--cond` should be set to 1, so that the training is done conditionally. In addition, the start and the end of the transition from unconditional to conditional training should be specified using the arguments `t_start_kimg` and `--t_end_kimg`. Here is an example training command:

```.bash
python train.py --outdir=./out/ \
--data=datasets/ImageNet_Carnivores_20_100.zip \
--cond=1 --t_start_kimg=2000  --t_end_kimg=4000  \
--gpus=4 \
--cfg=auto --mirror=1 \
--metrics=fid50k_full,kid50k_full
```

See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#training-new-networks) for more details on the arguments, configurations amd hyper-parammeters. Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list of arguments.

Note: Our code currently can be used only for unconditional or transitional training. For the original conditional training, you can use the original implementation StyleGAN2+ADA.

## Evaluation and Logging<a name="Evaluation"></a>

By default, `train.py` automatically computes FID for each network pickle exported during training. More metrics can be added to the argument `--metrics` (as a comma-seperated list).  To monitor the training, you can inspect the log.txt an JSON files (e.g. `metric-fid50k_full.jsonl` for FID) saved in the ouput directory.  Alternatively, you can inspect WandB or Tensorboard logs (By default, WandB creates the logs under the project name "Transitional-cGAN", which can be accessed in your account on the website). 

When desired, the automatic computation can be disabled with `--metrics=none` to speed up the training slightly (3%&ndash;9%). Additional metrics can also be computed after the training:

```.bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=pr50k3_full \
    --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

The first example looks up the training configuration and performs the same operation as if `--metrics=pr50k3_full` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--mirror` and `--data` must be specified explicitly.


See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#quality-metrics) for more details on the available metrics. 

## Contact<a name="Contact"></a>
For any questions, suggestions, or issues with the code, please contact Mohamad Shahbazi at [mshahbazi@vision.ee.ethz.ch](mailto:mshahbazi@vision.ee.ethz.ch)<br>


## How to Cite<a name="How-to-Cite"></a>

```
@inproceedings{
shahbazi2022collapse,
title={Collapse by Conditioning: Training Class-conditional {GAN}s with Limited Data},
author={Shahbazi, Mohamad and Danelljan, Martin and Pani Paudel, Danda and Van Gool, Luc},
booktitle={The Tenth International Conference on Learning Representations },
year={2022},
url={https://openreview.net/forum?id=7TZeCsNOUB_}
```
