#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
设置全局变量
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 200010
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 1
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.0005
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the lanenet training batch size
__C.TRAIN.BATCH_SIZE = 8
# Set the lanenet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 8
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 410000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 2
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 512


# set additional parameters for hnet
# Set the hnet training batch size
__C.TRAIN.BATCH_SIZE_HNET = 24
# Set the hnet validation batch size
__C.TRAIN.VAL_BATCH_SIZE_HNET = 24
# Set the image height for hnet
__C.TRAIN.IMG_HEIGHT_HNET = 64
# Set the image width for hnet
__C.TRAIN.IMG_WIDTH_HNET = 128
# Set max distance allowed for fitting
__C.TRAIN.MAX_DIST = 500.0
# Set learning rate for HNET
__C.TRAIN.LEARNING_RATE_HNET = 0.00001


# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 8


# Parameters related to Datasets
__C.DATASET = edict()
# Original image height from dataset
__C.DATASET.IMG_HEIGHT_ORG = 720
# Original image height from dataset
__C.DATASET.IMG_WIDTH_ORG = 1280
# Focal length in pixels
__C.DATASET.f = 1280
# Camera height in meters
__C.DATASET.Tz = 1.5
# Set the max num of lanes per image used for padding
__C.DATASET.MAX_NUM_LANE = 4
# Set the max num of samples per lane used for padding
__C.DATASET.MAX_NUM_LANE_SAMPLE = 56