import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import random
import pandas as pd
import indexToclass
import data_split
#临时配置torch环境变量
os.environ['TORCH_HOME'] = 'torch/hub/cache'
# 获取TORCH_HOME环境变量的值
torch_home = os.environ.get('TORCH_HOME')
print('当前缓存路径为:', torch_home)
data_split.data_split()
