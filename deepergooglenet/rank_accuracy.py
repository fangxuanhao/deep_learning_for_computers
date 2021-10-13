#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/9 上午10:26
# @Author : fangxuanhao
#导入必要的包
from deepergooglenet.config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.mean_preprocessor import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.oi.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import json

#加载rgb均值到训练集

means = json.loads(open(config.DATASET_MEAN).read())

#初始化图像预处理
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

#初始化测试数据生成器
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

#加载预先训练好的网络
print("[INFO] loading model")
model = load_model(config.MODEL_PATH)

#对测试数据预测
print("[INFO] predicting on test data...")
predictions = model.predict_generator(testGen.generator(), steps=testGen.num_images // 64, max_queue_size=64 * 2)

#computer the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, testGen.db['labels'])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 *100))

#关闭数据库
testGen.close()
