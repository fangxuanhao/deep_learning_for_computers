#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/24 下午3:06
# @Author : fangxuanhao
from deepergooglenet.config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.oi.hdf5datasewriter import HDF5DatasetWriter
from imutils import paths

import progressbar
import json
import cv2
import os
import numpy as np




#获取到训练图像的路径，然后提取训练 类标签并对它们进行编码

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLables = [p.split(os.path.sep)[-3] for p in trainPaths]

le = LabelEncoder()
trainLables = le.fit_transform(trainLables)
#从训练集分层抽样，构建测试集
split = train_test_split(trainPaths, trainLables, test_size=config.NUM_TEST_IMAGES, stratify=trainLables, random_state=42)
(trainPaths, testPaths, trainLables, testLables) = split

#从file中加载验证类filename =>，然后使用这些 构建验证路径和标签列表的映射
M = open(config.VAL_MAPPINGS).read().strip().split('\n')
# print(M)
M = [r.split("\t")[:2] for r in M]
# print(M)
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLables = le.transform([m[1] for m in M])

#构建匹配相对应的训练集测试集验证集路径标签和，HDF5文件
datasets = [
    ("train", trainPaths, trainLables, config.TRAIN_HDF5),
    ("val", valPaths, valLables, config.VAL_HDF5),
    ("test", testPaths, testLables, config.TEST_HDF5)
]

#初始化rgb通道的均值列表
(R, G, B) = ([], [], [])
for (dType, paths, labels, outputPath) in datasets:
    #创建HDF5 写入文件
    print("[INFO] building {}...".format(outputPath))
    # writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

    #初始化进度条
    # widgets = ["Building Dataset: ", progressbar.ProgressBar(), "", progressbar.Bar(), " ", progressbar.ETA()]
    # pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()  # 进度条

    #循环图像路径直到结束
    for (i, (path, label)) in enumerate(zip(paths, labels)):

        #从硬盘加载图像
        image = cv2.imread(path)

        # 如果我们正在构建训练数据集，那么计算图像中每个通道的平均值，然后更新各自的列表
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        #将图片标签加载到HDF5中
        # writer.add([image], [label])
        pbar.update()

    #关闭HDF5写入文件
    pbar.finish()
    # writer.close()

#构建一个平均值字典，将平均值序列化到json文件中
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")#必须绝对路径
f.write(json.dumps(D))
f.close()








