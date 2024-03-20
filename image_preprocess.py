from torchvision import transforms
##增强训练
reinforce_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),##随机地水平翻转图像
    transforms.RandomRotation(10),##随机地旋转图像
    transforms.RandomResizedCrop(224),##随机地裁剪图像
    transforms.RandomVerticalFlip(),##随机地垂直翻转图像
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),##改变图像的亮度、对比度和饱和度
    transforms.ToTensor(),##将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])##归一化
])
##未增强训练集，验证集，测试集
general_transform = transforms.Compose([
    transforms.Resize(256),##将图像的短边缩放到256像素
    transforms.CenterCrop(224),##从图像的中心裁剪出一个224x224像素的区域
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


