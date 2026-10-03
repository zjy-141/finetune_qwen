import torch
from unsloth import FastLanguageModel
import triton

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("可用:", torch.cuda.is_available())
print("能力:", torch.cuda.get_device_capability(0))
print("Unsloth OK")
print("Triton OK")