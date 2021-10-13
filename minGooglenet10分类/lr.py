#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/6 4:46 下午
# @Author : fangxuanhao

NUM_EPOCHS = 100
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

print(poly_decay(80))