#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/29 下午4:53
# @Author : fangxuanhao
# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

#import necessary packages
from dog_vs_cat.config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patch_preprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.mean_preprocessor import MeanPreprocessor
from pyimagesearch.oi.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# construct the training image generator for data augmentation

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15
                         , horizontal_flip=True, fill_mode="nearest")
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap]
                                , classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128, preprocessors=[sp, mp, iap]
                              , classes=2)
# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3) #默认 衰减速率 0。999 动量系数 0。9 手动调整的时候推荐用 sgd + 动量法 可以自控 缺点比adam要慢 一般先用adam初始化学习一下，在手动使用sgd+ 动量法在做调整
#或者先用动量法+sgd下降，在使用adam做最后的加速
'''
bn层经常插入到全连接层之后，非线性激活函数前

'''
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# construct the set of callbacks
path = os.path.sep.join(([config.OUTPUT_PATH, "{}.png".format(os.getpid())]))
callbacks = [TrainingMonitor(path)]

#train  the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.num_images // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.num_images // 128,
    epochs=75,
    max_queue_size=128 * 2,
    callbacks=callbacks,
    verbose=1
)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

#close the HDF5 datasets

trainGen.close()
valGen.close()
