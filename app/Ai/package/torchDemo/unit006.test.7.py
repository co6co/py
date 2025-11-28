#-*- coding: utf-8 -*-
# 激活函数的作用
import torch

# 创建随机张量
random_tensor=torch.randn(10) 
# 应用ReLU激活函数
relu_tensor=torch.relu(random_tensor)
print("random_tensor:",random_tensor)
print("relu_tensor:",relu_tensor)

# 应用Sigmoid激活函数
sigmoid_tensor=torch.sigmoid(random_tensor)
print("sigmoid_tensor:",sigmoid_tensor)

# 应用Tanh 激活函数
tanh_tensor=torch.tanh(random_tensor)
print("tanh_tensor:",tanh_tensor)

# 应用Softmax 激活函数
leaky_relu_tensor=torch.nn.LeakyReLU(random_tensor)
print("leaky_relu_tensor:",leaky_relu_tensor)
