'''
   images,labels =next(iter(train_loader))
   print(images.shape,labels)
   ##将tensor转为array
   images = images.numpy()
   ##绘图，显示该例图片的像素值分布
   plt.hist(images[2].flatten(),bins=50)
   plt.show()
   # batch 中经过预处理的图像
   idx = 0
   plt.imshow(images[idx].transpose((1, 2, 0)))  # 转为(224, 224, 3)
   plt.title('处理后图像label:' + str(labels[idx].item()))
   label = labels[idx].item()
   plt.show()
   ##加载npy文件
   data = np.load('idx_to_labels.npy', allow_pickle=True)
   idx_to_labels = data.item()
   pred_classname = idx_to_labels[label]
   # 原始图像
   idx = 0
   mean = np.array([0.485, 0.456, 0.406])
   std = np.array([0.229, 0.224, 0.225])
   testa = np.clip(images[idx].transpose((1, 2, 0)) * std + mean,0,1)
   print(testa)
   plt.imshow(np.clip(images[idx].transpose((1, 2, 0)) * std + mean,0,1))
   plt.title('原图像label:' + pred_classname)
   plt.show()
   '''

'''
   ##训练一个batch
   images, labels = next(iter(train_loader))
   images = images.to(device)
   labels = labels.to(device)
   outputs = vgg16(images)
   # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
   loss = criterion(outputs, labels)
   # 反向传播“三部曲”
   optimizer.zero_grad()  # 清除梯度
   loss.backward()  # 反向传播
   optimizer.step()  # 优化更新
   # 获得当前 batch 所有图像的预测类别
   _, preds = torch.max(outputs, 1)
   print(loss,preds)
   total = 0
   correct = 0
   total += labels.size(0)
   correct += (preds == labels).sum()  # 预测正确样本个数
   print('准确率为 {:.3f} %'.format(100 * correct / total))
   '''
'''
##遍历每个EPOCH
for epoch in tqdm(range(EPOCHS)):
    vgg16.train()  ##调整为训练模式
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(images)  ##前向预测
        loss = criterion(outputs, labels)
        optimizer.zero_grad()  ##清除梯度
        loss.backward()  ##反向传播
        optimizer.step()  ##更新优化

vgg16.eval()  ##切换为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(val_loader):  # 获取测试集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(images)  # 前向预测，获得当前 batch 的预测置信度
        _, preds = torch.max(outputs, 1)  # 获得最大置信度对应的类别，作为预测结果
        total += labels.size(0)
        correct += (preds == labels).sum()  # 预测正确样本个数
    print('验证集上的准确率为 {:.3f} %'.format(100 * correct / total))
torch.save(vgg16, 'torch/hub/checkpoints/vgg248(base_test)')
'''