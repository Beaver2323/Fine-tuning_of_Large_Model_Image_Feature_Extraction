from PIL import Image

# 打开图像
image = Image.open('COVID-19_Radiography_Dataset/COVID/images/COVID-1.png')

# 获取图像的模式
mode = image.mode

print(mode)