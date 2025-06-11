# LOSA
This project provides the implementation of the Learnable Online Style Adaptation (LOSA), which achieve state-of-the-art performance on the online test-time domain adaptive object detection task.

## Introduction
Please follow [DA_Detection](https://github.com/VisionLearningGroup/DA_Detection.git) respository to setup the environment. In this project, we use Pytorch 0.4.0. 

## Datasets
### Datasets Preparation
* **GTAV10k dataset:** Download our [GTAV10k](https://drive.google.com/file/d/1dlGy5L7ko_I8qdPRTWhK5wc1Z-VXfK7a/view) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).


### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Google Drive](https://drive.google.com/file/d/1KyZZi_GQq6x6PqO-3MKPC1OB5VlBIQx8/view?usp=sharing)
* **ResNet101:** [Google Drive](https://drive.google.com/file/d/1UuoXgslnA4Y-ZoyW0d2jViTkRl6HnHIC/view?usp=sharing)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Train
Source domain train:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
python  trainval_SFso.py   --cuda --lr 0.001  --net res101  --dataset gta_car --dataset_t  ucas_car   --save_dir training/SF
```
## Test
Source model test:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
test_SFso.py  --dataset ucas_car --net res101 --cuda --load_name training/SF/res101/gta_car/SF_source_False_target_ucas_car_gamma_5_1_3_9999.pth
```
and OTTAOD model test:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
python test_TTA_ours.py --dataset ucas_car --net res101 --cuda --load_name training/SF/res101/gta_car/SF_source_False_target_ucas_car_gamma_5_1_3_9999.pth
```
## Citation
@article{liu2025losa,<br>
  title={LOSA: Learnable Online Style Adaptation for Test-time Domain Adaptive Object Detection},<br>
  author={Liu, Weixing and Luo, Bin and Liu, Jun and Nie, Han and Su, Xin},<br>
  journal={IEEE Transactions on Geoscience and Remote Sensing},<br>
  number={99},<br>
  pages={1--1},<br>
  year={2025},<br>
  publisher={IEEE}<br>
}

