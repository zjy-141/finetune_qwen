import os
import argparse

# ============ 默认配置 ============
DEFAULT_HF_CACHE = r"D:\00_Inbox\huggingface_cache"
DEFAULT_MODEL = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit"
DEFAULT_DATA = r"D:\01_Project\project\2026-project18\code\data.jsonl"
DEFAULT_OUTPUT = r"D:\01_Project\project\2026-project18\code\outputs"
DEFAULT_LORA = r"D:\01_Project\project\2026-project18\code\lora_model"
# ==================================

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="训练数据 jsonl 路径")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="本地模型路径")
parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="训练输出目录")
parser.add_argument("--lora", type=str, default=DEFAULT_LORA, help="LoRA 保存目录")
parser.add_argument("--hf_cache", type=str, default=DEFAULT_HF_CACHE, help="HF 缓存目录")
parser.add_argument("--max_steps", type=int, default=200, help="训练步数")
parser.add_argument("--batch_size", type=int, default=2, help="批大小")
parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
args = parser.parse_args()

os.environ["HF_HOME"] = args.hf_cache

# 检查数据文件是否存在（提前报错，避免浪费模型加载时间）
if not os.path.exists(args.data):
    raise FileNotFoundError(f"数据文件不存在: {args.data}")

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. 加载数据集
dataset = load_dataset("json", data_files=args.data, split="train")
print(f"数据集加载成功，共 {len(dataset)} 条样本")

# 4. 数据格式化
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )

# 5. 配置训练参数
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    args=SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output,
    ),
)

# 6. 训练
trainer.train()

# 7. 保存
model.save_pretrained(args.lora)
tokenizer.save_pretrained(args.lora)
print(f"LoRA 适配器已保存到 {args.lora}")