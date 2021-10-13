#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/24 下午2:22
# @Author : fangxuanhao
#导入必要的包
from os import path

#定义训练集和验证集路径
TRAIN_IMAGES = "/Users/fangxuanhao/bdy/tiny-imagenet-200/train"
VAL_IMAGES = "/Users/fangxuanhao/bdy/tiny-imagenet-200/val/images"

#定义验证文件名映射到的文件的路径对应的类标签
VAL_MAPPINGS= "/Users/fangxuanhao/bdy/tiny-imagenet-200/val/val_annotations.txt"

#定义使用的WordNet层次结构文件的路径生成类标签
WORDNET_IDS = "/Users/fangxuanhao/bdy/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "/Users/fangxuanhao/bdy/tiny-imagenet-200/words.txt"

# #因为我们无法访问我们需要的测试数据 从训练数据中提取一些图像，并使用它代替（抽取测试集）
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

#定义输出的训练集测试集，验证集HDF5图片路径
TRAIN_HDF5 = "/Users/fangxuanhao/bdy/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "/Users/fangxuanhao/bdy/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "/Users/fangxuanhao/bdy/tiny-imagenet-200/hdf5/test.hdf5"

#定义数据集均值保存路径
DATASET_MEAN= "/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/deepergooglenet/output/tiny-image-net-200-mean.json" #求均值必须绝对路径

# 定义用于存储图的输出目录的路径分类报告等
OUTPUT_PATH = "output"
MODEL_PATH = ""
FIG_PATH = ""
JSON_PATH = ""
