import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import random
import pandas as pd
import indexToclass
#临时配置torch环境变量
os.environ['TORCH_HOME'] = 'torch/hub/cache'
# 获取TORCH_HOME环境变量的值
torch_home = os.environ.get('TORCH_HOME')
print('当前缓存路径为:',torch_home)

#指定数据集路径
dataset_path = 'dataset'
dataset_name = dataset_path
new_dataset_path = 'slipt_dataset'
print('使用的数据集:',dataset_name)
##提取数据集中的类别，输出类别数量
classes = os.listdir(dataset_path)
print('类别数:',len(classes),'分别为:',classes)
#创建“train”文件夹
os.mkdir(os.path.join(new_dataset_path,'train'))
os.mkdir(os.path.join(new_dataset_path,'val'))
os.mkdir(os.path.join(new_dataset_path,'test'))
##各集合的比例
test_frac = 0.2
train_frac = 0.6
val_frac=0.2
random.seed(123)##随机数种子,以便于复现
tab = pd.DataFrame()##新建一个数据结构，用于等会存储分类文件
print('{:^18} {:^18} {:^18}'.format('类别','训练集个数','验证集个数','测试集个数'))
for l_class in classes :##遍历每个x光片类别
    temp_dir = os.path.join(dataset_path,l_class)##各文件夹
    image_filename = os.listdir(temp_dir)##image_filename为当前文件夹内的图片集合
    random.shuffle(image_filename)##随机打乱文件名
    ##划分训练集和测试集
    train_num = int(len(image_filename)*train_frac)
    val_num = int(len(image_filename)*val_frac)
    train_data = image_filename[:train_num] ##数据集的前train_num个元素作为训练集
    val_data = image_filename[train_num:train_num + val_num] ##
    test_data = image_filename[train_num + val_num:]
    ##切片操作左闭右开！！！！！！！！不会产生交集
    ##生成三种集合
    for image in train_data:
        temp_img_path = os.path.join(new_dataset_path, l_class, image)  # 获取原始文件路径
        new_img_path = os.path.join(new_dataset_path, 'train', l_class,image)  # 获取 test 目录的新文件路径
        shutil.move(temp_img_path,new_img_path)  # 移动文件
    for image in val_data:
        temp_img_path = os.path.join(new_dataset_path, l_class, image)  # 获取原始文件路径
        new_img_path = os.path.join(new_dataset_path, 'val', l_class,image)  # 获取 test 目录的新文件路径
        shutil.move(temp_img_path,new_img_path)  # 移动文件
    for image in test_data:
        temp_img_path = os.path.join(new_dataset_path, l_class, image)  # 获取原始文件路径
        new_img_path = os.path.join(new_dataset_path, 'test', l_class,image)  # 获取 test 目录的新文件路径
        shutil.move(temp_img_path,new_img_path)  # 移动文件
    '''
    ##删除原有文件夹
    assert len(os.listdir(temp_dir)) == 0  # 确保旧文件夹中的所有图像都被移动走
    shutil.rmtree(temp_dir)  # 删除文件夹
    '''
    ##输出每一类别的数据个数
    print('{:^18} {:^18} {:^18} {:^18}'.format(l_class, len(train_data), len(val_data)),len(test_data))
    ##保存到表格中
    tab = tab.append({'class': l_class, 'train_num': len(train_data), 'val_num': len(val_data), 'test_num':len(test_data) },ignore_index=True)
    tab.to_csv