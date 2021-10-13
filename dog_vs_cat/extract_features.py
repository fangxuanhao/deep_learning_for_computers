#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/29 下午4:53
# @Author : fangxuanhao

#基于resnet 50 提取特征

#import the neccessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img
from pyimagesearch.oi.hdf5datasewriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", default='/Users/fangxuanhao/zqy/Kaggle猫狗大战/train', help="path to input dataset")
ap.add_argument("-o", "--output", default='hdf5/features.hdf5', help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=16, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())

#store the batch size in a convenience variable
bs = args["batch_size"]

#获取我们要描述的图片随机洗牌，以便训练期间的数组切片
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

#extract the class labels form the image paths then cncode the labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)
#load the resnet 50 network
print('[INFO] loading network...')
model = ResNet50(weights='imagenet', include_top=False)
#初始化hdf5 数据集然后写入标签 和数据
dataset = HDF5DatasetWriter((len(imagePaths), 2048),
                            args["output"],
                            data_key="features",
                            buf_size=args["buffer_size"]
                            )
dataset.store_class_labels(le.classes_)

#初始化进度条
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
widgets=widgets).start()

#批量循环图像
'''
1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
'''
for i in np.arange(0, len(imagePaths), bs):
    '''
    提取一批图像和标签，然后初始化 将通过网络的实际图像列表用于特征提取
    '''
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    #循环当前处理的标签图像
    for (j, imagePaths) in enumerate(batchPaths):
        #加载图像同时确保图像大小为 224*224
        print(imagePaths)
        image = load_img(imagePaths, target_size=(224, 224))
        image = img_to_array(image)
        '''
        #预处理图像(1)扩大尺寸和 76 #(2)减去平均RGB像素强度 # ImageNet数据集
        '''
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        #将图像加入批量处理中
        batchImages.append(image)

    #通过将图像输入网络提取特征
    batchImages = np.vstack(batchImages) #垂直方向堆叠数组组成新的数组
    features = model.predict(batchImages, batch_size=bs)
    print(features.shape)

    '''
    #重塑特征，使每个图像都可以用 ‘MaxPooling2D’输出的一个扁平特征向量
    '''
    features = features.reshape((features.shape[0],7*7* 2048))
    dataset.add(features, batchLabels)
    pbar.update(i)
#ValueError: cannot reshape array of size 1605632 into shape (16,2048)

#关闭数据集
dataset.close()
pbar.finish()





