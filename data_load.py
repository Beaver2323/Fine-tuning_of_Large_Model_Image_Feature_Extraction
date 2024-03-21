import os
from torchvision import datasets
import image_preprocess
##配置三集合路径

def data_load(dataset_path):
    ##dataset_path = 'slipt_dataset'
    train_path = os.path.join(dataset_path,'train')
    val_path = os.path.join(dataset_path,'val')
    test_path = os.path.join(dataset_path,'test')
    print('训练集路径:',train_path,'验证集路径:',val_path,'测试集路径:',test_path)
    train_dataset = datasets.ImageFolder(train_path,image_preprocess.general_transform)
    val_dataset = datasets.ImageFolder(val_path,image_preprocess.general_transform)
    test_dataset = datasets.ImageFolder(val_path, image_preprocess.general_transform)
    print('训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)
    print('验证集图像数量', len(val_dataset))
    print('类别个数', len(val_dataset.classes))
    print('各类别名称', val_dataset.classes)
    print('测试集图像数量', len(test_dataset))
    print('类别个数', len(test_dataset.classes))
    print('各类别名称', test_dataset.classes)
    return train_dataset,val_dataset,test_dataset,len(train_dataset.classes)