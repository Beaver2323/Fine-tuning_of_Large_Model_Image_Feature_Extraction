import os


def temp_torch_home():
    #临时配置torch环境变量
    os.environ['TORCH_HOME'] = 'torch/hub/cache'
    # 获取TORCH_HOME环境变量的值
    torch_home = os.environ.get('TORCH_HOME')
    print('当前缓存路径为:', torch_home)