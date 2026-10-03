# finetune_qwen

在单张 8GB 笔记本显卡上，用 **Unsloth + 4-bit QLoRA** 对 **Qwen3-4B-Instruct-2507** 做中文闲聊（LCCC）微调，并把 LoRA 适配器导出为 **GGUF (Q4_K_M)**，最终部署到 WSL 中的 **Ollama**。

- 仓库：`git@github.com:zjy-141/finetune_qwen.git`
- 详细分步操作：见 [操作指南.md](操作指南.md)（本 README 只做总览与快速上手）
- 当前进度：**模型已下载完成，尚未开始训练**

## 目录

- [1. 目标与路线](#1-目标与路线)
- [2. 软硬件环境](#2-软硬件环境)
- [3. 目录结构](#3-目录结构)
- [4. 快速上手](#4-快速上手)
- [5. 脚本说明](#5-脚本说明)
- [6. 数据说明](#6-数据说明)
- [7. 关键参数](#7-关键参数)
- [8. 常见问题](#8-常见问题)
- [9. 注意事项](#9-注意事项)

## 1. 目标与路线

整条流水线刻意拆成**互不耦合的四段**，每段都能单独重跑，失败时不必从头再来：

```
下载模型  →  准备数据  →  LoRA 微调  →  导出 GGUF  →  Ollama 部署
（一次性）   （复制即可）  （可反复跑）   （可反复跑）    （WSL）
```

| 阶段 | 入口 | 产物 | 状态 |
| :--- | :--- | :--- | :--- |
| ① 下载模型 | `code/download_model.py` | `models/Qwen3-4B-Instruct-bnb-4bit/`（约 3.4GB） | ✅ 已完成 |
| ② 准备数据 | `code/data.jsonl` | 4.0 MB 训练数据 | ⬜ 待做 |
| ③ LoRA 微调 | `code/finetune_qwen.py` | `code/lora_model/`、`code/outputs/` | ⬜ 待做 |
| ④ 导出 GGUF | `code/export_gguf.py`（**待补写**） | `code/model_gguf/unsloth.Q4_K_M.gguf` | ⬜ 待做 |
| ⑤ 部署 | WSL 中的 `ollama create` | `qwen-chat` 模型 | ⬜ 待做 |

## 2. 软硬件环境

| 项目 | 版本 / 配置 |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB，驱动 592.01） |
| 计算能力 | `(12, 0)`，即 Blackwell / sm_120，**必须用 cu128 及以上的 PyTorch** |
| 系统 | Windows + Anaconda Prompt（WSL 用于最后的 Ollama 部署） |
| Conda 环境 | `unsloth311`（Python 3.11，路径 `D:\05_DevTools\Programme\anaconda\envs\unsloth311`） |
| PyTorch | 2.9.1+cu128（torchvision 0.24.1+cu128、torchaudio 2.9.1+cu128） |
| Unsloth | 2026.9.14（unsloth_zoo 2026.9.9） |
| xformers | 0.0.33.post2 |
| Triton | triton-windows 3.8.0.post29 |
| 训练框架 | transformers 5.5.0、trl 0.24.0、datasets 4.3.0、peft 0.21.2、bitsandbytes 0.50.2 |

环境搭建的完整终端记录保留在 [log/unsloth.txt](log/unsloth.txt)，其中包含两次典型的踩坑与修复：

1. `pip install unsloth` 会把 torch 升到 2.12.1，导致与 cu128 组合错位 → 需卸载后重装 `torch==2.9.1+cu128`。
2. 随后 `xformers 0.0.35` 要求 `torch>=2.10` → 用 `pip install xformers==0.0.33.post2 --no-deps` 降级解决。

### 激活并自检

```cmd
conda activate unsloth311
cd /d D:\01_Project\project\2026-project18\code
python test_3.py
```

预期输出：

```
torch: 2.9.1+cu128
CUDA: 12.8
可用: True
能力: (12, 0)
Unsloth OK
Triton OK
```

任何一项失败，先看 [操作指南.md](操作指南.md) 第 7 节故障排查。

## 3. 目录结构

```
2026-project18/
├─ README.md                     ← 本文件
├─ 操作指南.md                    ← 完整分步指南（含全部代码与排错表）
├─ .gitignore
├─ .vscode/settings.json         ← 固定使用 conda 环境
├─ .obsidian/                    ← Obsidian 笔记配置（已忽略）
├─ code/
│  ├─ download_model.py          ← ① 下载模型到本地目录
│  ├─ finetune_qwen.py           ← ② 4-bit QLoRA 微调
│  ├─ export_gguf.py             ← ③ 导出 GGUF（尚未创建）
│  ├─ test_torch.py              ← 环境自检：torch / CUDA / 矩阵运算
│  ├─ test_unsloth.py            ← 环境自检：Unsloth
│  ├─ test_triton.py             ← 环境自检：Triton
│  ├─ test_3.py                  ← 上面三项的合并版
│  ├─ test_dataset.py            ← 数据自检：打印首行 JSON 结构
│  ├─ data.jsonl                 ← 训练数据（待生成，已被忽略）
│  ├─ lora_model/                ← 微调产物：LoRA 适配器（待生成）
│  ├─ outputs/                   ← 微调产物：日志与检查点（待生成）
│  ├─ model_gguf/                ← 导出产物：GGUF（待生成）
│  └─ unsloth_compiled_cache/    ← Unsloth 自动生成，可删可重建
├─ dataset/                      ← LCCC 中文闲聊数据（已忽略）
│  ├─ LCCC-base-close/
│  │  ├─ LCCC-base_train.jsonl   （1390.7 MB）
│  │  ├─ LCCC-base_valid.jsonl   （4.0 MB，本次使用）
│  │  └─ LCCC-base_test.jsonl    （2.0 MB）
│  └─ LCCC-base_test.jsonl       （与上一行重复，可删）
├─ models/
│  └─ Qwen3-4B-Instruct-bnb-4bit/  ← 已下载模型（约 3.4GB，已忽略）
└─ log/
   └─ unsloth.txt                ← 环境搭建终端记录（已忽略）
```

> `.gitignore` 已忽略 `models/`、`log/`、`dataset/**/*.jsonl`、`unsloth_compiled_cache/`、`*.gguf`、`*.safetensors`，因此仓库里**只有代码与文档**，大文件不会入库。

## 4. 快速上手

全程在 **Anaconda Prompt** 中操作，命令都在 `code` 目录下执行（脚本里的相对路径依赖当前工作目录）。

```cmd
conda activate unsloth311
cd /d D:\01_Project\project\2026-project18\code
```

### 步骤 1：下载模型（已完成，换机时重跑）

```cmd
python download_model.py
```

- 源：`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- 目标：`models\Qwen3-4B-Instruct-bnb-4bit`（约 3.4GB，含权重与 tokenizer 共 12 个文件）
- 元数据缓存：`D:\00_Inbox\huggingface_cache`
- 直连超时：取消脚本中 `HF_ENDPOINT = "https://hf-mirror.com"` 那一行的注释

### 步骤 2：准备训练数据

```cmd
copy "D:\01_Project\project\2026-project18\dataset\LCCC-base-close\LCCC-base_valid.jsonl" data.jsonl
python test_dataset.py
```

格式应为 `{"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}`。若 `test_dataset.py` 读的是 test 文件，把里面的路径改成 `data.jsonl` 即可。

### 步骤 3：开始微调

```cmd
python finetune_qwen.py
```

- 从本地模型目录加载，**不会联网**
- 200 步，单步日志实时打印，正常 loss 应降到 1.5 以下
- 8GB 显存吃紧时：`per_device_train_batch_size` 改 1、`gradient_accumulation_steps` 改 8
- 产物：`lora_model\`（适配器）、`outputs\`（日志与检查点）

### 步骤 4：导出 GGUF

先按 [操作指南.md](操作指南.md) 第 5 节补写 `export_gguf.py`，再运行：

```cmd
python export_gguf.py
```

得到 `model_gguf\unsloth.Q4_K_M.gguf`。

> 输出目录将要自定义：`output_dir`、`lora_model`、`model_gguf` 目前都是相对路径，改成绝对路径后，请同步更新第 5.2 节的运行目录与第 6.1 节的复制来源。

### 步骤 5：部署到 Ollama

在 WSL 中：

```bash
mkdir -p ~/qwen-chat
cp /mnt/d/01_Project/project/2026-project18/code/model_gguf/unsloth.Q4_K_M.gguf ~/qwen-chat/
cd ~/qwen-chat
ollama create qwen-chat -f Modelfile
ollama run qwen-chat
```

`Modelfile` 的完整内容（ChatML 模板 + 采样参数）见 [操作指南.md](操作指南.md) 第 6.2 节。

## 5. 脚本说明

### 业务脚本

| 脚本 | 作用 | 关键输入 | 关键输出 |
| :--- | :--- | :--- | :--- |
| `download_model.py` | 从 Hugging Face 拉取 4-bit 量化基座模型到本地目录 | 仓库 ID、`local_dir` | `models\Qwen3-4B-Instruct-bnb-4bit\` |
| `finetune_qwen.py` | 加载本地模型 → 注入 LoRA → 读 `data.jsonl` → 训练并保存适配器 | `data.jsonl` | `lora_model\`、`outputs\` |
| `export_gguf.py` | 基座 + LoRA 合并后导出 Q4_K_M 的 GGUF | `lora_model\` | `model_gguf\unsloth.Q4_K_M.gguf` |

三个脚本都显式设置同一个 `HF_HOME`（`D:\00_Inbox\huggingface_cache`），**改缓存目录时三处要一起改**。

### 自检脚本（非流程必需，可随时单独运行）

| 脚本 | 检查内容 |
| :--- | :--- |
| `test_torch.py` | torch 版本、CUDA 版本、是否可用、计算能力、GPU 矩阵运算 |
| `test_unsloth.py` | Unsloth 能否导入、CUDA 是否正常 |
| `test_triton.py` | Triton 能否导入 |
| `test_3.py` | 以上三项合并，一次跑完 |
| `test_dataset.py` | 打印数据首行的 JSON 结构，验证 `conversations` 字段 |

## 6. 数据说明

- 数据来源：LCCC-base（close 版）中文闲聊，已转换为 `conversations` 多轮对话格式。
- 本次训练使用 **验证集** `LCCC-base-close\LCCC-base_valid.jsonl`（4.0 MB），只跑 200 步以验证整条链路是否跑通。
- 仓库被 `.gitignore` 忽略，因此 `data.jsonl` 和 `dataset\` 都不会入库；换机后需从原项目 `2026-project17` 或数据备份恢复。
- `dataset\LCCC-base_test.jsonl` 与 `dataset\LCCC-base-close\LCCC-base_test.jsonl` 内容完全相同，前者是多余副本，可删除以省 2 MB。

## 7. 关键参数

| 项目 | 取值 | 说明 |
| :--- | :--- | :--- |
| 基座模型 | Qwen3-4B-Instruct-2507（4-bit 量化） | 非 thinking 模式，不输出 `<think>` 块 |
| 微调方式 | LoRA，`r=16`，`lora_alpha=16`，`lora_dropout=0` | QLoRA 4-bit 加载，8GB 显存的关键 |
| 目标模块 | `q/k/v/o_proj` + `gate/up/down_proj` | 注意力 + MLP 全覆盖 |
| 最大序列长度 | 2048 | 可通过 `max_seq_length` 调整 |
| 批大小 | 2 × 梯度累积 4 = 等效 8 | 显存不足时改 1 × 8 |
| 学习率 / 步数 | `2e-4` / 200 步 | 线性衰减，warmup 5 步 |
| 优化器 | `adamw_8bit`，`weight_decay=0.01` | 8-bit 优化器省显存 |
| 精度 | 按硬件自动选 bf16 / fp16 | 代码里用 `torch.cuda.is_bf16_supported()` 判断 |
| 随机种子 | 3407 | 训练与 LoRA 初始化一致 |
| GGUF 量化 | Q4_K_M | 精度与体积的平衡点 |
| 模型体积 | 约 3.4GB | `model.safetensors` 单个 3381.7 MB |

## 8. 常见问题

| 现象 | 处理 |
| :--- | :--- |
| `CUDA out of memory` | `per_device_train_batch_size=1` + `gradient_accumulation_steps=8`；仍不够则把 `max_seq_length` 降到 1024 |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch，用 cu128 版本重装 |
| `pip install unsloth` 后 CUDA 失效 | 它会把 torch 升级，需按 [log/unsloth.txt](log/unsloth.txt) 里的步骤重装 torch 2.9.1+cu128 与 xformers 0.0.33.post2 |
| `ModuleNotFoundError: No module named 'triton'` | 装 `triton-windows`（`pip install unsloth` 会带上），并确认在 `unsloth311` 环境中 |
| 训练 loss 不降 | 用 `test_dataset.py` 确认 `conversations` 字段正确；可把学习率提到 3e-4 |
| 下载模型卡住 / 超时 | 在下载脚本里启用 `HF_ENDPOINT = "https://hf-mirror.com"` |
| 模型输出乱码或重复 | Ollama 的 `Modelfile` 模板必须与训练时的 ChatML 模板一致 |
| `export_gguf.py` 报文件不存在 | `export_gguf.py` 尚未创建，需先按操作指南第 5 节补齐 |
| 找不到 `lora_model` / 检查点落错位置 | 脚本用的是相对路径，必须先 `cd` 到 `code` 目录再运行 |

## 9. 注意事项

- **必须以 `code` 为工作目录运行脚本**，否则 `data.jsonl`、`lora_model`、`outputs`、`model_gguf` 这类相对路径会落错位置。
- **三个脚本的 `HF_HOME` 必须一致**（当前为 `D:\00_Inbox\huggingface_cache`），否则会重复下载。
- **sm_120 显卡对版本敏感**：torch 必须是 cu128 构建，xformers 必须与 torch 匹配，升级任意一个前先在 `log/unsloth.txt` 的口径下确认兼容性。
- **大文件不入库**：模型、数据、GGUF 都被 `.gitignore` 忽略，克隆仓库只得到代码和文档；换机后需重新下载/复制数据。
- 磁盘占用参考：`models\` 约 3.4GB、`dataset\` 约 1.3GB、训练中间产物另计。
