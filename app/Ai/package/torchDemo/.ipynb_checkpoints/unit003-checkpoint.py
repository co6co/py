# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 10:20
# 自动微分

# 线性一阶导数
import torch
w=torch.tensor([1.0],requires_grad=True) # 创建需要梯度的张量w
x=torch.tensor([2.0],requires_grad=True) # 创建需要梯度的张量x
a=torch.add(w,x) # 计算a=w+x
b=torch.add(w,1)
y=torch.mul(a,b) # 计算y=a*b

y.backward() # 对y进行反向传播，计算梯度
print("w.grad:",w.grad) # 输出w的梯度 w.grad: tensor([5.])
print("x.grad:",x.grad) # 输出x的梯度 x.grad: tensor([2.])


# grad_tensors 参数
y0=torch.mul(a,a) 
y1=torch.add(a,b) 
loss=torch.cat([y0,y1],dim=0) # 拼接y和y1，dim=0表示按行拼接，形成新的张量loss
grad_t=torch.tensor([1.0,2.0]) # 创建梯度张量grad_tensors，用于指定每个元素的梯度权重
loss.backward(gradient=grad_t ) # 对loss进行反向传播，计算梯度，指定梯度权重为grad_tensors
print("w.grad:",w.grad) # 输出w的梯度 w.grad: tensor([15.]) 
print("x.grad:",x.grad) # 输出x的梯度 x.grad: tensor([10.])


# 求取梯度
x=torch.tensor([3.],requires_grad=True) # 创建需要梯度的张量x
y=torch.pow(x,2) # 计算y=x^2
grad1=torch.autograd.grad(y,x,create_graph=True) # 计算y关于x的梯度，参数 create_graph 表示同时计算二阶导数
print("grad1:",grad1) # 输出grad1: (tensor([6.]),)
grad2=torch.autograd.grad(grad1[0],x) # 计算grad1[0]关于x的梯度
print("grad2:",grad2) # 输出grad2: (tensor([2.]),)
 
## 注意事项
# 1. 梯度不能自动清零，在每次反向传播中会叠加
w=torch.tensor([1.0],requires_grad=True) # 创建需要梯度的张量w
x=torch.tensor([2.0],requires_grad=True) # 创建需要梯度的张量x
for i in range(3):
    a=torch.add(x,w)
    b=torch.add(w,1)
    y=torch.mul(a,b)
    y.backward()
    print("w=>",w.grad) 
    #输出分别是：
    #tensor([5.])
    #tensor([10.])
    #tensor([15.])
    print("x=>",x.grad) 
    # 2. 每次反向传播后，需要手动清零梯度
    w.grad.zero_()
    x.grad.zero_()

# 2. 依赖于叶子节点的节点，requires_grad默认为True
w=torch.tensor([1.0],requires_grad=True) # 创建需要梯度的张量w
x=torch.tensor([2.0],requires_grad=True) # 创建需要梯度的张量x
a=torch.add(w,x) # 计算a=w+x

b=torch.add(w,1)
y=torch.mul(a,b)
print("a,b,y是否需要梯度：",a.requires_grad,b.requires_grad,y.requires_grad)

# 3. 叶子节点不可以执行 in-place,因为向前传播记录了叶子节点地址，反向传播需要用到叶子节点的数据时，
#要根据地址寻找数据，执行in-place操作改变了地址中的数据，梯度求解会发生错误
w=torch.tensor([1.0],requires_grad=True) # 创建需要梯度的张量w
x=torch.tensor([2.0],requires_grad=True) # 创建需要梯度的张量x
a=torch.add(w,x) # 计算a=w+x

b=torch.add(w,1)
y=torch.mul(a,b)
try:
    w.add_(1) # 对w进行原地加法操作，即w=w+1
except RuntimeError as e:
    print("Error:",e)

a=torch.tensor([1.0] )   
print("a初始地址:",id(a),a)
a=a+torch.tensor([1.0])
print("开辟新地址：",id(a),a)
# in-place 操作，地址不变
a+=torch.tensor([1.0])
print("a+=操作的地址：",id(a),a)