# celeba-baselines

## Overview

This repository contains code for training a simple ResNet-18 model on the CelebA dataset. Both the cropped/aligned images and the original, "in-the-wild" images are supported. While simple, this model achieves results remarkably similar to current state-of-the-art. We also provide evaluation code which provides accuracy/balanced accuracy/f1 metrics which can be averaged over multiple runs.

Please note that the CelebA dataset is highly imbalanced and exhibits a large degree of bias (see ["Covering CelebA Bias
With Markov Blankets"](https://yapdianang.github.io/celeba/) for a helpful reference). Models trained on this dataset are therefore not suitable for real-world use and should be limited to usage in academic research.

## File Inventory

- `celeba.py`: pytorch dataset class for CelebA.
- `celeba_resnet_train.py`: code for training a ResNet-18 model on CelebA.
- `celeba_evaluate.py`: code for collection evaluation metrics of a trained ResNet-18 model on CelebA.

## Requirements

We list the specific version of each requirement we use but later versions and some earlier versions of most libraries are likely to work as well.
- python 3.7.3. Any version >=3.6 should be fine.
- pytorch 1.5.1, torchvision 0.6. Any pytorch version >=1.1.0 should be fine.
- Cuda 10.1. Training code should work on CPU but we have not tested this and would not recommend it. Evaluation code works on CPU but is slow.
- kornia 0.3.1. Later versions of Kornia generate warning messages, though as far as we can tell this does not impact results.
- numpy 1.18.1. Most versions should work fine.

## Usage

Training a model requires the CelebA dataset, which can be acquired [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). For the aligned images, the folder layout should look like this:
```
CelebA
├── images
│   ├── 000001.jpg
│   ├── 000002.jpg
...
│   └── 202599.jpg
└── labels
    └── list_attr_celeba.txt
```
For running the code with the original/uncropped images, you should have a separate folder titled `img_celeba` containing all images.

The training code assumes that both folders are located in the current directory. If this is not the case, you can modify the global path constants on lines 17 and 18 of `celeba_resnet_train.py`. We found it easier to modify these constants rather than needing to specify the directory every time the code is run. The training code takes the following arguments:

- `-b`, `--batch-size`: controls the batch size. Default: `256`
- `-l`, `--lr`: controls the learning rate. Default: `.1`.
- `-e`, `--epochs`: number of epochs to train. Default: `40`.
- `-d`, `--device`: device to use for training. Default: `cuda:0`.
- `-p`, `--pretrain`: use weights pretrained on imagenet. Default: `False`.
- `-a`, `--aligned`: use aligned version of CelebA. Default: `True`.
- `-a`, `--aligned`: use aligned version of CelebA. Default: `True`.
- `-s`, `--scheduler`: learning rate scheduler to use. Can be either `multiplicative` or `plateau` Default: `multiplicative`.
- `--multiplier`: multiplier to use for multiplicative learning rate schedule. Has no effect if another scheduler is used. Default: `.9`.
- `--patience`: patience parameter to use for lr reduction on plateau. Has no effect if another scheduler is used. Default: `10`.
- `--factor`: factor parameter to use for lr reduction on plateau. Has no effect if another scheduler is used. Default: `.1`.
- `-r`, `--sample-rate`: rate to downsample the dataset by. Will make a new numpy file in ./samples/ unless one already exists. Default: `1`.
- `-c`, `--split-idx`: index of custom splits file (note used in the paper). Should be located in ./new-splits/. Default: `None`.
- `-i`, `--id`: string to append to model name / log file. Default: None.
- `-w`, `--weigh-loss`: Weigh attribute losses using ratio between negative and positive examples in training set. Default: `False`.
- `-v`, `--val-rate`: If set, will only compute and print vaidation results every val_rate epochs. Note that the lr reduction on plateau schedule requires validation results to be computed every epoch.

The following configurations can be used to replicate our results:
- Cropped and aligned: `python celeba_resnet_train.py`
- Cropped and aligned, pretrained: `python celeba_resnet_train.py -p -e 20 --multiplier .8`
- Cropped and aligned, 10%: `python celeba_resnet_train.py -e 80 -s plateau -r 10`
- Cropped and aligned, 10%, pretrained: `python celeba_resnet_train.py -p -r 10`
- Uncropped: `python celeba_resnet_train.py -a -l .025 -b 64 -s plateau -e 60`
- Uncropped, pretrained: `python celeba_resnet_train.py -a -p -l .025 -b 64 -e 20`
- Uncropped, 10%: `python celeba_resnet_train.py -a -l .025 -b 64 -e 80 -s plateau -r 10`
- Uncropped, 10%, pretrained: `python celeba_resnet_train.py -a -p -l .025 -b 64 -r 10`

We provide results which are averaged over several runs by giving each training run a unique identifier. This can be done using the `-i` argument. For example, run one should have `-i 1`, run two should have `-i 2`, etc. Our results from running the above commands are provided in the results folder. These results were used for all tables and figures.

`celeba_evaluate.py` is used to compute validation and testing results. To compute validation results for a single model, use: `python celeba_evaluate.py MODEL_PATH -s val`. To compute mean and standard deviation of 5 different runs, use `python celeba_evaluate.py MODEL_PATH_BASE --average -s val`. The model names are assumed to be `MODEL_PATH_BASE_1`, `MODEL_PATH_BASE_2`, ..., `MODEL_PATH_BASE_5`, as would be generated by `-i 1`, `-i 2`, ... `-i 5`. For test results, simply replace `val` in the command with `test`.
