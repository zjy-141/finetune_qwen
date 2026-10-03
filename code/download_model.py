import os

# 设置缓存目录（模型下载到这里）
os.environ["HF_HOME"] = r"D:\00_Inbox\huggingface_cache"

# 如果 Hugging Face 直连超时，取消下面这行的注释启用镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 下载到本地指定目录
local_dir = r"D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit"

snapshot_download(
    repo_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"模型已下载到: {local_dir}")