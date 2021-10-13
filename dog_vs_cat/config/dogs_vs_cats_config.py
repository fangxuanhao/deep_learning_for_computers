#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/29 下午4:51
# @Author : fangxuanhao
IMAGES_PATH = "/Users/fangxuanhao/zqy/Kaggle猫狗大战/train"
# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
#分别一个抽取2500张图像用于测试和验证
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files

TRAIN_HDF5 = "/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/dog_vs_cat/hdf5/train.hdf5"
# f = open('/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/dog_vs_cat/hdf5/train.hdf5',"w")
VAL_HDF5 = "/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/dog_vs_cat/hdf5/val.hdf5"
TEST_HDF5 = "/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/dog_vs_cat/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

# define the path to the dataset mean
DATASET_MEAN = "/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/deep learn/dog_vs_cat/output/dogs_vs_cats_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"