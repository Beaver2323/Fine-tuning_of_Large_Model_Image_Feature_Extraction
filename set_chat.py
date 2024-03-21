import matplotlib.pyplot as plt

def set_chat():
    ##显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']
    ##用于正常显示负号
    plt.rcParams['axes.unicode_minus']=False