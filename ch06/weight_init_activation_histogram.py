import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # 1000条数据，(1000, 100)矩阵
node_num = 100  # 每个隐藏层中的节点（神经元）数量
hidden_layer_size = 5  # 5个隐藏层
activations = {}  # 在此处存储激活结果

x = input_data

for i in range(hidden_layer_size):  # 遍历5个隐藏层
    if i != 0:
        x = activations[i - 1]

    # 通过更改初始值的各种值来实验
    # w = np.random.randn(node_num, node_num) * 1  # (100, 100) 矩阵
    # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    a = np.dot(x, w)  # (1000, 100)

    # 尝试不同类型的激活函数
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z  # 上一层的输出作为本层的输入

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + '-layer')
    if i != 0:
        plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
