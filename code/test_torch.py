# 检查 CUDA 是否可用，以及 GPU 的名称和计算能力，并进行简单的矩阵运算测试

import torch
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("可用:", torch.cuda.is_available())
print("能力:", torch.cuda.get_device_capability(0))
x = torch.randn(100, 100, device="cuda")
print("运算:", torch.mm(x, x).sum().item())