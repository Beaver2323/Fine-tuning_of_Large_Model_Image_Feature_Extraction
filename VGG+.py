import torch
from torchvision import models, transforms
from PIL import Image
import os
import indexToclass
#临时配置torch环境变量
os.environ['TORCH_HOME'] = 'torch/hub/cache'
# 获取TORCH_HOME环境变量的值
torch_home = os.environ.get('TORCH_HOME')
print('当前缓存路径为:',torch_home)

vgg16 = models.vgg16(weights = False)
weights = torch.load('torch/hub/checkpoints/vgg16-397923af.pth')
vgg16.load_state_dict(weights)
vgg16.eval()  # set the model to evaluation mode

img_path = 'COVID-19_Radiography_Dataset/COVID/images/COVID-1.png'  # replace with the path to your image
img = Image.open(img_path)
img = img.convert('RGB')
mode = img.mode
##print(mode)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    vgg16.to('cuda')

with torch.no_grad():
    output = vgg16(input_batch)

_, predicted_idx = torch.max(output, 1)
print('该物品被预测为:', indexToclass.idx_to_class[predicted_idx.item()])
