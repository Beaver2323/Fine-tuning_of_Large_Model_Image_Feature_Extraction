import numpy as np


def generate_index(set):
    idx_to_labels = {y: x for x, y in set.class_to_idx.items()}
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', set.class_to_idx)
    '''
    # 使用numpy.load函数打开.npy文件
    data = np.load('labels_to_idx.npy', allow_pickle=True)
    labels_to_idx = data.item()
    # 现在，data是一个NumPy数组，你可以像正常的NumPy数组一样使用它
    print(labels_to_idx)
    # 使用numpy.load函数打开.npy文件
    data = np.load('idx_to_labels.npy', allow_pickle=True)
    idx_to_labels = data.item()
    # 现在，data是一个NumPy数组，你可以像正常的NumPy数组一样使用它
    print(idx_to_labels)
    '''
    print('索引已生成')