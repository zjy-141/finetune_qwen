from unsloth import FastLanguageModel
import torch

print("Unsloth OK")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Capability: {torch.cuda.get_device_capability(0)}")