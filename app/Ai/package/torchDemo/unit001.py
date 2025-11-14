# 查看 pytorch 环境是否安装好
# (AI) pip install torch torchvision torchaudio
import torch 
print("pytorch Version:",torch.version.__version__)
# 创建一个 2x3 的随机张量
x=torch.rand(2,3)
print("tensor shape:",x.shape)

print("tensor dtype:",x.dtype)
# 
Y=torch.rand(3,2)
z=torch.matmul(x,Y)
print("matmul result:",z)
print("z shape:",z.shape)