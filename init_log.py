import pandas as pd
from train_val import train_one_batch
from train_val import val
# 初始化训练日志
def init_train_log(train_loader,device,vgg16,criterion,optimizer,epoch,batch_idx):
    tab_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images,labels = next(iter(train_loader))##未开始循环训练前的空白性能
    log_train.update(train_one_batch(images, labels,device,vgg16,criterion,optimizer,epoch,batch_idx))
    tab_train_log = tab_train_log._append(log_train, ignore_index=True)
    return log_train,tab_train_log
# 初始化验证日志
def init_val_log(val_loader,device,model,criterion,epoch):
    tab_val_log = pd.DataFrame()
    log_val = {}
    log_val['epoch'] = 0
    log_val.update(val(val_loader,device,model,criterion,epoch))
    tab_val_log = tab_val_log._append(log_val, ignore_index=True)
    return log_val,tab_val_log