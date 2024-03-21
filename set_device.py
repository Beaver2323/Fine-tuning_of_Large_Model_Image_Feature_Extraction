import torch
def set_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用的计算硬件是',device)
    return device