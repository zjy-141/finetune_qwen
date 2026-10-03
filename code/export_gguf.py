import os

os.environ["HF_HOME"] = r"D:\00_Inbox\huggingface_cache"

from unsloth import FastLanguageModel

MODEL_PATH = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit"

# 加载基础模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 加载 LoRA 适配器
model.load_adapter("lora_model")

# 导出为 GGUF（Q4_K_M 量化，适合 8GB 显存推理）
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF 导出完成，查看 model_gguf 文件夹。")