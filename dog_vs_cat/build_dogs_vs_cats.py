#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/29 下午4:52
# @Author : fangxuanhao
# define the paths to the images directory

# import the necessary packages
from dog_vs_cat.config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspect_aware_preprocessor import AspectAwarePreprocessor
from pyimagesearch.oi.hdf5datasewriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLables = [p.split(os.path.sep)[-1].split('.')[0] for p in trainPaths]

le = LabelEncoder()
trainLables = le.fit_transform(trainLables)

split = train_test_split(trainPaths, trainLables, test_size=config.NUM_TEST_IMAGES, stratify=trainLables, random_state=42)
(trainPaths, testPaths, trainLables, testLables) = split

split = train_test_split(trainPaths, trainLables, test_size=config.NUM_VAL_IMAGES, stratify=trainLables, random_state=42)
(trainPaths, valPaths, trainLables, valLables) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files

datasets = [('train', trainPaths, trainLables, config.TRAIN_HDF5),
            ('test', testPaths, testLables, config.TEST_HDF5),
            ('val', valPaths, valLables, config.VAL_HDF5)]

# initialize the image preprocessor and the lists of RGB channel 初始化rgb 通道 求平均
# averages #更能反应出图像的特点  目标检测，图像分类 常用

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

#循环遍历datasets元祖

for (dType, paths, lables, outputPath) in datasets:
    # os.makedirs('./hdf5/train.hdf5', exist_ok=True)
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    # print(outputPath)

    # writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets = widgets).start() #进度条
    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, lables)):
        # load the image and process it
        image = cv2.imread(path)

        image = aap.preprocess(image)
        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
            # add the image and label # to the HDF5 dataset
        # writer.add([image], [label])
        pbar.update(i)
    pbar.finish()
    # writer.close()
# construct a dictionary of averages, then serialize the means to a
# JSON file #构建平均值字典，放入json文件中
print("[INFO] serializing means...")
#求均值
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()










