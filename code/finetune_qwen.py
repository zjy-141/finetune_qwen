import os

# 设置 Hugging Face 缓存目录（保持和下载时一致）
os.environ["HF_HOME"] = r"D:\00_Inbox\huggingface_cache"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 本地模型路径
MODEL_PATH = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit"

# 1. 从本地加载模型与 Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
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
dataset = load_dataset("json", data_files="data.jsonl", split="train")

# 4. 将 conversations 转为纯文本
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=200,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# 6. 开始训练
trainer.train()

# 7. 保存 LoRA 适配器
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("LoRA 适配器已保存到 ./lora_model")