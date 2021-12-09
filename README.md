# DPPN-GZSL
PyTorch implementation for NeurIPS2021 paper: Dual Progressive Prototype Network for Generalized Zero-Shot Learning.
In this paper, we handle the critical issue of domain shift problem, i.e., confusion between seen and unseen categories, by progressively improving cross-domain transferability and category discriminability of visual representations.
![moti](https://user-images.githubusercontent.com/58110770/144179674-b4a64a16-23e0-4c81-be7e-d02d92f02e80.png)

# Quick Start

## Install

- Install PyTorch 1.2.
- Install dependencies: pip install -r requirements.txt


## Data and Model Preparation
Please download CUB, SUN, aPY datasets, and ResNet101 pretrained model.

## Train

Please specify the script file. 
For example, if you want to train on CUB dataset, you should specify the DATAPATH  (for dataset path), SAVEPATH (for log and model save dir path), RESNETPRE (for resnet pretrained model path) in the scripts/cub.sh, and then run on three 1080ti GPUs:

``` 
cd scripts
bash cub.sh
``` 




## Test
You can evaluate our pretrained model (https://drive.google.com/drive/folders/1uvtTO5o4xp_bXV2txiYLGPmfJQuM7lwt?usp=sharing) or your own model.

Please specify the DATAPATH  (for dataset path), SAVEPATH (for log and model save dir path), MODELPATH (for the test model) in the script file, and then run:

``` 
cd scripts
bash cub_test.sh
``` 

# Citation

If you find this work or code is helpful in your research, please cite:

``` 
@inproceedings{
wang2021dual,
title={Dual Progressive Prototype Network for Generalized Zero-Shot Learning},
author={Chaoqun Wang and Shaobo Min and Xuejin Chen and Xiaoyan Sun and Houqiang Li},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021}
}
``` 
