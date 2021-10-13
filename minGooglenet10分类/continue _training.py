#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/6 4:58 下午
# @Author : fangxuanhao


# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/23 下午3:53
# @Author : fangxuanhao
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from pyimagesearch.nn.conv.minigooglenet import MiniGoogLeNet  # 导入迷你googlenet
from keras.callbacks import ModelCheckpoint  # 保存每个阶段模型
from keras.preprocessing.image import ImageDataGenerator  # 图像增强
from keras.datasets import cifar10  # 导入cifar10分类
from keras.optimizers import SGD  # 导入 随机梯度下降优化器
from keras.callbacks import TensorBoard  # tensroboard 可视化
from keras.callbacks import CSVLogger  # 保存模型训练信息
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor  # 导入实时绘图工具和保存训练损失值
from keras.callbacks import LearningRateScheduler  # 导入学习速率调节器
import numpy as np
import argparse
import os
from keras.models import load_model

NUM_EPOCHS = 70
INIT_LR = 5e-3
BATCH_SIZE = 32


def poly_decay(epoch):  # 多项式衰减
    # 初始化最大纪元，基础学习率和多项式幂
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # 根据多项式衰减计算新的学习率

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # 返回新的学习率
    return alpha


# 构造实参并解释实参
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", default="checkpoints", help="path to output modle")
ap.add_argument("-o", "--output", default="output", help="path to output directory(logs, plots, etc)")  # etc 等等
ap.add_argument("-l", "--logs", default="logs", help="path to logs")
ap.add_argument('-i', '--initial-epoch', type=int, default=5, help='Epoch at which to start training')
ap.add_argument('-m', '--model', type=str, default='checkpoints/weights-005.h5', help='which epoch of the model will be loaded')

args = vars(ap.parse_args())

# 加载训练集和测试集，转换图片从整数到浮点
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# 对图像使用均值像素减法
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# 将标签转为向量
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# 构造用于数据图像增强器
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

os.makedirs(args['checkpoints'], exist_ok=True)
os.makedirs(args['logs'], exist_ok=True)
os.makedirs(args['output'], exist_ok=True)

fname = os.path.join(args['checkpoints'], "weights-{epoch:03d}.h5")
# 构造回调函数
fig_path = os.path.join(args['output'], "{}.png".format(os.getpid()))
json_path = os.path.join(args['output'], "{}.json".format(os.getpid()))
checkpoint = ModelCheckpoint(fname, mode='min', save_best_only=False, verbose=1)
'''
Modlecheckpoint fname(字符串，保存模型的路径)
mode = mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
'''
tensorboard = TensorBoard(log_dir=args['logs'])
csv_Log = CSVLogger(filename='logs/training.csv')
tr_mo = TrainingMonitor(fig_path, json_path)
lr_sheduler = LearningRateScheduler(poly_decay)  # 跟新自定义学习率


def train_model(model, trainX, trainY, testX, testY, initial_epoch):
    # 整合回调函数
    callbacks = [
        checkpoint, tensorboard, csv_Log, lr_sheduler, tr_mo
    ]

    # 初始化优化器和模型

    '''
    #monentum 动量，模拟物体运动时的惯性，即跟新的时候一定程度上保持之前的跟新方向，同时利用当前的batch的梯度微调最终的跟新方向，
    # 这样可以在一定程度上增加稳定性，从而学习的更快，并且还有一定的摆脱局部最优的能力
    '''

    # 训练网络
    print("[INFO] training network...")
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks, verbose=1,
        initial_epoch=initial_epoch
    )
    # verbose 打印出进度条

    print(H)
    print(H.history)

if __name__ == '__main__':

    # 编译模型
    if args['model'] is None:
        model = MiniGoogLeNet(width=32, height=32, depth=3, classes=10)
        opt = SGD(lr=INIT_LR, momentum=.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])  # metrics 评判标准 准确率
        model.summary()
    else:
        model = load_model(args['model'])
    train_model(model, trainX, trainY, testX, testY, initial_epoch = args['initial_epoch'])





'''
1. 先验知识。在阅读了数百篇深度学习论文、博客文章、教程之后
更不用说，进行你自己的实验，你会开始注意到
一些数据集。在我的例子中，我从职业生涯中使用CIFAR10数据集的先前实验中了解到，50-100个时期之间的任何时间通常都是训练所需的全部时间
CIFAR-10。网络架构越深（具有足够的正则化），以及
学习率的下降，通常会让我们的网络训练时间更长。因此，
我选择70个时代作为我的第一个实验。实验结束后，我
可以检查学习情节并决定是否应该使用更多/更少的时代（随着时间的推移）
出局时，70轮）。
2不可避免的过度装配。第二，从本书之前的实验中我们知道
当使用CIFAR-10时，我们最终会过度拟合。这是不可避免的；即使有强大的
正则化和数据扩充，仍然会发生。因此，我决定70轮
而不是冒着80-100轮的风险，过度拟合的影响会变得更加严重明显的。
'''
# 学习率调高了容易过拟合
