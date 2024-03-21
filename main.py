import os
from tqdm import tqdm
##设置内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import temp_torch_home
import set_device
import set_chat
import data_load
import generate_index
import init_log
import warnings
import wandb
from train_val import train_one_batch
from train_val import val


warnings.filterwarnings("ignore")

def main():
    ##配置缓存路径
    temp_torch_home.temp_torch_home()
    ##忽略烦人的红色提示
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ##可视化图表
    set_chat.set_chat()
    ##设置计算硬件
    device = set_device.set_device()
    print('使用的计算硬件是', device)
    ##载入图像分类数据集
    train_dataset,val_dataset,test_dataset,class_num = data_load.data_load('little_slipt_dataset')
    ##创建映射索引
    generate_index.generate_index(train_dataset)

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    ##载入预训练模型
    vgg16 = models.vgg16(weights=False)
    weights = torch.load('torch/hub/checkpoints/vgg16-397923af.pth')
    vgg16.load_state_dict(weights)##加载预训练权重
    vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, class_num)##替换全连接层
    optimizer = optim.Adam(vgg16.classifier[6].parameters())##只训练最后一层
    ##训练设置
    vgg16 = vgg16.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练轮次 Epoch
    EPOCHS = 20
    ##学习率下降策略
    lr_scheduler_1 = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    ##训练前设置
    epoch = 0
    batch_idx = 0
    best_val_accuracy = 0##用于迭代
    ##初始化训练日志和验证日志
    log_train,tab_train_log = init_log.init_train_log(train_loader,device,vgg16,criterion,optimizer,epoch,batch_idx)
    log_val, tab_val_log = init_log.init_val_log(val_loader,device,vgg16,criterion,epoch)
    ##训练可视化
    wandb.init(project='VGG+', name='248a(base_test_2)')
    ##循环训练
    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        vgg16.train()
        for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels,device,vgg16,criterion,optimizer,epoch,batch_idx)
            tab_train_log = tab_train_log._append(log_train, ignore_index=True)
            wandb.log(log_train)
        lr_scheduler_1.step() ##学习率下降策略
        ## 测试阶段
        vgg16.eval()
        log_val = val(val_loader,device,vgg16,criterion,epoch)
        tab_val_log = tab_val_log._append(log_val, ignore_index=True)
        wandb.log(log_val)

        # 保存最新的最佳模型文件
        if log_val['val_accuracy'] > best_val_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'torch/hub/checkpoints/best-{:.4f}.pth'.format(best_val_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_val_accuracy = log_val['val_accuracy']
            new_best_checkpoint_path = 'torch/hub/checkpoints/best-{:.4f}.pth'.format(log_val['val_accuracy'])
            torch.save(vgg16, new_best_checkpoint_path)
            print('保存新的最佳模型', 'checkpoint/best-{:.4f}.pth'.format(best_val_accuracy))
            # best_val_accuracy = log_test['val_accuracy']


    tab_train_log.to_csv('log/训练日志248a.csv', index=False)
    tab_val_log.to_csv('log/验证日志248a.csv', index=False)

if __name__ == '__main__':
    main()