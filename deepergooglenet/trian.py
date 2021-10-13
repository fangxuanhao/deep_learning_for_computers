#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/8 下午2:36
# @Author : fangxuanhao
import  matplotlib
matplotlib.use("Agg")

#导入必要的包
from deepergooglenet.config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.mean_preprocessor import MeanPreprocessor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint # 保存模型
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor# 画图
from pyimagesearch.oi.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.deepergooglenet import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import json
import os



#构造实体参数并解析实惨
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, default="checkpoints", help="path to output checkpoints directory")
ap.add_argument("-m", "--model", type=str, help="paths to *specific* model checkpoints to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch tp restart training at")
args = vars(ap.parse_args())

os.makedirs(args['checkpoints'], exist_ok=True)


#构建用于训练增强的数据增强器
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True
                         , fill_mode='nearest')
#加载rgb均值到训练集中
means = json.loads(open(config.DATASET_MEAN).read())

#初始化图片预处理
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

#初始化训练集和验证集生成器
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
val_Gen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

#如果没有特定的模型检查点，那就初始化网络和编译模型
if args['model'] is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
    opt = Adam(1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#否则就从磁盘上加载
else:
    print("[INFO] loding {}...".format(args["model"]))
    model = load_model(args["model"])
    #更新学习率
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, start_at=args['start_epoch']),
    TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH,
                    start_at=args["start_epoch"])
    ]

#训练网络
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.num_images // 64,
    validation_data=val_Gen.generator(),
    validation_steps=val_Gen.num_images // 64,
    epochs=10,
    max_queue_size=64 * 2,
    callbacks = callbacks, verbose=1
)

#关闭数据集
trainGen.close()
val_Gen.close()
