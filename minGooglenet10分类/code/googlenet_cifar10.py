#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/2 下午4:39
# @Author : fangxuanhao
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
#import the necessary packages 必要的
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.mingoogle import MiniGoogLeNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os
import cv2

#define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 10
INIT_LR = 5e-3


def poly_decay(epoch):
    #initailaize the maximum number of epochs base,laeaning rate and power of the polynomial
    maxEpochs  = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    #compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 -(epoch / float(maxEpochs))) ** power

    #return the new learning rate
    return alpha

#construct the argument parse and parse the arguments  构造实参并解析实参
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="model/minigooglenet_cifar10.h5", help="path to output model")
ap.add_argument("-o", "--output", default="output", help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())
#load the training and testing data, converting the images from integers to floats
print("[INFO] loading CIFAR-10 data... ")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

#apply mean subtraciton to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

#convert the labels from integers to vectors convert 转换  vectors 向量
lb = LabelBinarizer()
trainY= lb.fit_transform(trainY)
testy = lb.fit_transform(testY)
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

#consturuct the set of callbacks
figPath = os.path.sep.join([args['output'], "{}.png".format(os.getpid())]) #返回当前进程id
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, json_path= jsonPath), LearningRateScheduler(poly_decay)]
#initialize the opetimizer and model
print("[INFO] compiling model... ")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the network
print("[INFO] training network")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64, epochs=NUM_EPOCHS
                    ,callbacks= callbacks, verbose=1)
print(H)
print(H.history)
#save the network to disk
print("[INFO] serializing network... ")
model.save(args['model'])





