# PEFT库的merge_and_unload()方法在底层会自动将4-bit基座解量化到全精度（float32/16-bit)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径配置
BASE_MODEL_PATH = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-16bit"
LORA_PATH = r"D:\01_Project\project\2026-project18\lora_model"
MERGED_OUTPUT = r"D:\01_Project\project\2026-project18\merged_16bit_model"

print("加载16-bit基座模型...")
# 以16-bit精度加载基座（如果显存不足，可尝试 load_in_8bit=True）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用bfloat16以节省显存并保持精度
    device_map="auto",           # 自动分配显存和内存
    trust_remote_code=True,
)

print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("正在合并LoRA到16-bit基座 (merge_and_unload)...")
# merge_and_unload 会自动处理解量化与合并
merged_model = model.merge_and_unload()

print(f"保存合并后的16-bit模型到: {MERGED_OUTPUT}")
merged_model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
# 同时保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_OUTPUT)

print("完成！")