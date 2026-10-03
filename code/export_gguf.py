# 好像没用到
import os
import argparse

# ============ 可配置区 ============
DEFAULT_HF_CACHE = r"D:\00_Inbox\huggingface_cache"
DEFAULT_LORA = r"D:\01_Project\project\2026-project18\lora_model"
DEFAULT_GGUF_OUTPUT = r"D:\01_Project\project\2026-project18\gguf_output"
# ==================================

parser = argparse.ArgumentParser()
parser.add_argument("--lora", type=str, default=DEFAULT_LORA, help="LoRA 适配器路径")
parser.add_argument("--output", type=str, default=DEFAULT_GGUF_OUTPUT, help="GGUF 输出目录")
parser.add_argument("--hf_cache", type=str, default=DEFAULT_HF_CACHE, help="HF 缓存目录")
args = parser.parse_args()

os.environ["HF_HOME"] = args.hf_cache

# 检查 LoRA 目录
if not os.path.exists(args.lora):
    raise FileNotFoundError(f"LoRA 适配器目录不存在: {args.lora}")

adapter_config = os.path.join(args.lora, "adapter_config.json")
if not os.path.exists(adapter_config):
    raise FileNotFoundError(f"缺少 adapter_config.json: {adapter_config}")

os.makedirs(args.output, exist_ok=True)
print(f"LoRA 适配器: {args.lora}")
print(f"GGUF 输出目录: {args.output}")

from unsloth import FastLanguageModel

# ⚠️ 关键改动：直接加载 LoRA 适配器目录，而不是先加载基座模型再 load_adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.lora,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print(f"已加载微调模型（含 LoRA）")

# 导出 GGUF（Unsloth 会自动合并 LoRA 到基座模型再量化）
model.save_pretrained_gguf(
    args.output,
    tokenizer,
    quantization_method="q4_k_m",
    save_method="merged_4bit_forced", 
)

print(f"GGUF 导出完成: {args.output}")