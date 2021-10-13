#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/10/13 2:48 下午
# @Author : fangxuanhao
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspect_aware_preprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='/Users/fangxuanhao/fxh/方轩豪/workbase/Limingming/Deep-Learning-For-Computer-Vision-master/datasets/flowers17', help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation  图像增强器
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

'''
旋转 | 反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;
翻转变换(flip): 沿着水平或者垂直方向翻转图像;
缩放变换(zoom): 按照一定的比例放大或者缩小图像;
平移变换(shift): 在图像平面上对图像以一定方式进行平移;可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;
尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;
对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;
噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
错切变换（shear）：效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移，且平移的大小和该点到x轴(或y轴)的垂直距离成正比。
'''

opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64,height=64,depth=3,classes=len(classNames))
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

# train the network
print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX,trainY,batch_size=32),
# validation_data=(testX,testY), steps_per_epoch=len(trainX) // 32 ,epochs=100, verbose=1)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('minivggnet_flowers17_data_aug.png')
plt.show()

'''
              precision    recall  f1-score   support #原数据类别个数

    bluebell       0.59      0.80      0.68        20
   buttercup       0.65      0.87      0.74        15
  colts_foot       0.60      0.83      0.70        18
     cowslip       0.47      0.56      0.51        16
      crocus       0.56      0.43      0.49        21
    daffodil       0.62      0.35      0.44        23
       daisy       0.84      0.91      0.87        23
   dandelion       0.75      0.68      0.71        22
  fritillary       0.94      0.79      0.86        19
        iris       0.88      0.79      0.83        19
 lily_valley       0.52      0.61      0.56        18
       pansy       1.00      0.55      0.71        20
    snowdrop       0.38      0.65      0.48        20
   sunflower       0.92      1.00      0.96        23
   tigerlily       1.00      0.81      0.90        27
       tulip       0.47      0.44      0.45        16
  windflower       0.79      0.55      0.65        20

    accuracy                           0.69       340
   macro avg       0.70      0.68      0.68       340
weighted avg       0.72      0.69      0.69       340
'''



