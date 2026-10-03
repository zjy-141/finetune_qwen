import os
os.environ["HF_HOME"] = r"D:\00_Inbox\huggingface_cache"

from unsloth import FastLanguageModel

MODEL_PATH = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit"
LORA_PATH = r"D:\01_Project\project\2026-project18\lora_model"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model.load_adapter(LORA_PATH)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "你好啊"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))