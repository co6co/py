import torch
data=torch.manual_seed(123)
print(data)
data2=torch.initial_seed()
print(data2)
rng_state=torch.get_rng_state()
print("随机生成器状态->",rng_state)
