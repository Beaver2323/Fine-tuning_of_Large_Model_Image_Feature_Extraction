
import time
import os

import numpy as np
from tqdm import tqdm
from torchvision import  transforms


import torch

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import temp_torch_home
import set_device
import set_chat
import data_load
import generate_index
'''
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")
'''
##配置缓存路径
temp_torch_home.temp_torch_home()
##可视化图表
set_chat.set_chat()
##设置计算硬件
set_device.set_device()
##载入图像分类数据集
train_dataset,val_dataset,test_dataset = data_load.data_load()
##创建映射索引
generate_index.generate_index(train_dataset)
