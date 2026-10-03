# finetune_qwen

在单张 8GB 笔记本显卡上，用 **Unsloth + 4-bit QLoRA** 对 **Qwen3-4B-Instruct-2507** 做中文闲聊（LCCC）微调，再把 LoRA 合并到 16-bit 基座、经 **llama.cpp** 转成 **GGUF**，最终部署到 WSL 中的 **Ollama**。

- 仓库：`git@github.com:zjy-141/finetune_qwen.git`
- 详细分步操作：见 [操作指南.md](操作指南.md)（本 README 只做总览与快速上手）
- 当前进度：**已跑通"训练 → 合并 16-bit → 转 f16 GGUF"整条链路；已按你的要求把训练数据切到清洗后的训练集（671 万条）；待做——用训练集重训、量化 Q4_K_M、部署 Ollama**

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

整条流水线拆成**互不耦合的六段**，每段都能单独重跑，失败时不必从头再来：

```
下载基座 → 下载16bit基座 → 准备/清洗数据 → LoRA 微调 → 合并16bit → 转GGUF → 量化 → Ollama
（4bit）    （用于合并）     （clean_data）  （可反复跑） （peft）  （llama.cpp）（可选） （WSL）
```

| 阶段 | 入口 | 产物 | 状态 |
| :--- | :--- | :--- | :--- |
| ① 下载 4-bit 基座 | `hf download unsloth/…-bnb-4bit` | `models/Qwen3-4B-Instruct-bnb-4bit/` | ✅ 已完成 |
| ② 下载 16-bit 基座 | `hf download Qwen/Qwen3-4B-Instruct-2507` | `models/Qwen3-4B-Instruct-16bit/` | ✅ 已完成 |
| ③ 清洗数据 | `code/clean_data.py` | `dataset/LCCC-base_train_clean.jsonl`（1298.9 MB，671 万条）+ `…_valid_clean.jsonl`（3.84 MB） | ✅ 已完成 |
| ④ LoRA 微调 | `code/finetune_qwen.py`（默认读训练集、2000 步） | `lora_model/`、`outputs/` | ⚠️ 只有旧的一次 200 步产物，**需用训练集重训** |
| ⑤ 合并 16-bit | `code/merge_16bit_peft.py` | `merged_16bit_model/`（7.5 GB） | ✅ 已完成（基于旧 LoRA） |
| ⑥ 转 GGUF | `llama.cpp/convert_hf_to_gguf.py` | `gguf_output/qwen3_lccc_f16.gguf`（7.5 GB） | ✅ 已完成 |
| ⑦ 量化 Q4_K_M | `llama-quantize.exe` | `gguf_output/qwen3_lccc_q4_k_m.gguf`（约 2.5 GB） | ⬜ 待做（可选但推荐） |
| ⑧ 部署 | WSL 中的 `ollama create` | `qwen3-lccc` 模型 | ⬜ 待做 |

> **因果链提醒**：⑤⑥⑦ 的作品都基于 `lora_model/` 里当时的那一版适配器。一旦按新配置重训（④），`merged_16bit_model/`、`gguf_output/` 里的文件就都过期了，必须重跑 ⑤⑥⑦。
>
> **为什么要绕道 16-bit 合并**：Unsloth 的 `save_pretrained_gguf` 在 NF4 基座上无法直接做 16-bit 合并，会报错（[code/export_gguf.py](code/export_gguf.py) 是这条备选路线，当前未采用）。

## 2. 软硬件环境

| 项目 | 版本 / 配置 |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB，驱动 592.01） |
| 计算能力 | `(12, 0)`，即 Blackwell / sm_120，**必须用 cu128 及以上的 PyTorch** |
| 系统 | Windows + Anaconda Prompt（WSL 用于最后的 Ollama 部署） |
| Conda 环境 | `unsloth311`（Python 3.11，路径 `D:\05_DevTools\Programme\anaconda\envs\unsloth311`） |
| PyTorch | 2.9.1+cu128（torchvision 0.24.1+cu128、torchaudio 2.9.1+cu128） |
| Unsloth | 2026.9.14（unsloth_zoo 2026.9.9） |
| xformers / Triton | 0.0.33.post2 / triton-windows 3.8.0.post29 |
| 训练框架 | transformers 5.5.0、trl 0.24.0、datasets 4.3.0、bitsandbytes 0.50.2 |
| 合并阶段依赖 | `torchao==0.15.0`、`peft<0.16`（合并完若要继续训练，需把 `peft` 装回 `>=0.18.0`） |
| 转换/量化工具 | `2026-project17\llama.cpp`（转换脚本）、`2026-project17\llama-cpp-bin`（`llama-quantize.exe`） |

环境搭建的完整终端记录保留在 [log/unsloth.txt](log/unsloth.txt)，其中包含两次典型的踩坑与修复：

1. `pip install unsloth` 会把 torch 升到 2.12.1，导致与 cu128 组合错位 → 需卸载后重装 `torch==2.9.1+cu128`。
2. 随后 `xformers 0.0.35` 要求 `torch>=2.10` → 用 `pip install xformers==0.0.33.post2 --no-deps` 降级解决。

### 激活并自检

```cmd
conda activate unsloth311
cd /d D:\01_Project\project\2026-project18
python code\test_3.py
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

任何一项失败，先看 [操作指南.md](操作指南.md) 第 9 节常见问题。

## 3. 目录结构

```
2026-project18/
├─ README.md                     ← 本文件
├─ 操作指南.md                    ← 完整分步指南（含全部代码与排错表）
├─ .gitignore
├─ .vscode/settings.json         ← 固定使用 conda 环境
├─ .obsidian/                    ← Obsidian 笔记配置（已忽略）
├─ code/                         ← 所有脚本都放这里
│  ├─ clean_data.py              ← ③ 数据清洗/校验（生成 _clean.jsonl）
│  ├─ finetune_qwen.py           ← ④ 4-bit QLoRA 微调（argparse 参数化）
│  ├─ merge_16bit_peft.py        ← ⑤ LoRA 合并到 16-bit 基座
│  ├─ export_gguf.py             ← 备选：Unsloth 直接导出 GGUF（未采用）
│  ├─ model_test.py              ← 加载基座 + 适配器做一次推理自测
│  ├─ download_model.py          ← 原始下载脚本（现多用 hf download 命令）
│  ├─ test_torch.py / test_unsloth.py / test_triton.py / test_3.py
│  ├─ test_dataset.py            ← 数据自检：打印首行 JSON 结构
│  └─ unsloth_compiled_cache/    ← Unsloth 自动生成，可删可重建
├─ lora_model/                   ← 微调产物：LoRA 适配器（137 MB，已忽略）
│  ├─ adapter_config.json
│  ├─ adapter_model.safetensors
│  └─ tokenizer.json 等
├─ outputs/                      ← 微调产物：日志与检查点（约 200 MB，已忽略）
│  └─ checkpoint-*/              ← 含 optimizer.pt / trainer_state.json
├─ merged_16bit_model/           ← ⑤ 合并产物：16-bit 全量模型（7.5 GB，已忽略）
├─ gguf_output/                  ← ⑥⑦ 转换与量化产物（已忽略）
│  └─ qwen3_lccc_f16.gguf        （7.5 GB；q4_k_m 待生成）
├─ dataset/                      ← LCCC 中文闲聊数据（已忽略）
│  ├─ LCCC-base_train_clean.jsonl   （1298.9 MB，671 万条 ← 当前训练用）
│  ├─ LCCC-base_valid_clean.jsonl   （3.84 MB，19986 条）
│  ├─ LCCC-base_test.jsonl          （2.0 MB，与下面的 test 重复，可删）
│  └─ LCCC-base-close/              ← 原始数据放子目录
│     ├─ LCCC-base_train.jsonl   （1326.2 MB，682 万条，清洗的输入）
│     ├─ LCCC-base_valid.jsonl   （3.9 MB，2 万条）
│     └─ LCCC-base_test.jsonl    （2.0 MB，1 万条）
├─ models/                       ← 两个基座（共约 14.2 GB，已忽略）
│  ├─ Qwen3-4B-Instruct-bnb-4bit/  ← 训练用 4-bit
│  └─ Qwen3-4B-Instruct-16bit/     ← 合并用 16-bit
└─ log/                          ← 终端记录（已忽略）
   ├─ unsloth.txt                ← 环境搭建
   └─ finetune_test.txt          ← 首次 200 步训练输出
```

> 大文件与产物（`models/`、`merged_16bit_model/`、`gguf_output/`、`lora_model/`、`outputs/`、`log/`、`dataset/**/*.jsonl`、`*.gguf`、`*.safetensors`）都在 `.gitignore` 里，因此仓库里**只有代码与文档**。

## 4. 快速上手

在 **Anaconda Prompt** 中操作，命令统一在**项目根目录**执行（多数脚本用绝对路径，但保持同一口径最省心）：

```cmd
conda activate unsloth311
cd /d D:\01_Project\project\2026-project18
```

### 步骤 1-2：下载两个基座（已完成）

```cmd
hf download unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit --local-dir D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-bnb-4bit
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir D:\01_Project\project\2026-project18\models\Qwen3-4B-Instruct-16bit
```

- 4-bit 版用于训练（约 3.4GB），16-bit 版用于合并（约 10.7GB）。
- 中断后可用镜像续传：`set HF_ENDPOINT=https://hf-mirror.com` 后再跑同一命令。

### 步骤 3：清洗数据（已完成，可重跑）

```cmd
python code\clean_data.py
```

- 输入 `dataset\LCCC-base-close\LCCC-base_valid.jsonl` → 输出 `dataset\LCCC-base_valid_clean.jsonl`（**清洗产物放 dataset 根目录**，原始数据留在 `LCCC-base-close\` 子目录）
- 训练集同理：`--input "…\LCCC-base-close\LCCC-base_train.jsonl" --output "…\dataset\LCCC-base_train_clean.jsonl"`，实测保留 6716583 / 6820506（98.48%），仅剔除 103923 条重复
- 大文件建议先用 `--dryrun` 预演（不写文件、几秒出结果）
- 常用参数：`--max_chars 8192`、`--no_dedup`、`--dryrun`

### 步骤 4：LoRA 微调

```cmd
python code\finetune_qwen.py
```

- 默认读 **`dataset\LCCC-base_train_clean.jsonl`（671 万条）**、2000 步（详见 [5. 脚本说明](#5-脚本说明) 的参数表）
- 步数与数据量要配对看：`batch_size=2` × `grad_accum=4` = 每步 8 条，2000 步 ≈ 1.6 万条 ≈ **0.24 个 epoch**；想跑满一个 epoch 需约 84 万步
- 从本地模型目录加载，**不会联网**；产物固定落在项目根目录的 `lora_model\`、`outputs\`
- 8GB 显存吃紧时：`--batch_size 1 --grad_accum 8`
- 旧版 200 步的实测参考：257.9 秒 / `train_loss = 3.366`，仅 0.16 epoch（见 [log/finetune_test.txt](log/finetune_test.txt)）
- 训练后用 `python code\model_test.py` 让基座 + 适配器生成一句，验证效果

### 步骤 5：合并 LoRA 到 16-bit 基座

先装合并阶段需要的版本（见 [操作指南.md](操作指南.md) 第 1.4 节）：

```cmd
pip install torchao==0.15.0
pip install "peft<0.16"

python code\merge_16bit_peft.py
```

得到 `merged_16bit_model\`（约 7.5 GB）。合并完若要继续训练，记得 `pip install "peft>=0.18.0"` 装回去。

### 步骤 6：转换成 f16 GGUF

```cmd
cd /d D:\01_Project\project\2026-project17\llama.cpp
python convert_hf_to_gguf.py "D:\01_Project\project\2026-project18\merged_16bit_model" --outfile "D:\01_Project\project\2026-project18\gguf_output\qwen3_lccc_f16.gguf" --outtype f16
```

约 7.5 GB；转换时的 RoPE 警告无害，可忽略。

### 步骤 7：量化成 Q4_K_M（可选但推荐）

```cmd
cd /d D:\01_Project\project\2026-project17\llama-cpp-bin
llama-quantize.exe "D:\01_Project\project\2026-project18\gguf_output\qwen3_lccc_f16.gguf" "D:\01_Project\project\2026-project18\gguf_output\qwen3_lccc_q4_k_m.gguf" Q4_K_M
```

约 2.5 GB，显存占用大幅下降，推理更快。

### 步骤 8：部署到 Ollama

在 WSL 中：

```bash
mkdir -p ~/qwen-chat
cp /mnt/d/01_Project/project/2026-project18/gguf_output/qwen3_lccc_q4_k_m.gguf ~/qwen-chat/
cd ~/qwen-chat
nano Modelfile
ollama create qwen3-lccc -f Modelfile
ollama run qwen3-lccc
```

不想量化也行，把 `Modelfile` 的 `FROM` 改成 `./qwen3_lccc_f16.gguf` 即可。`Modelfile` 完整内容（含 ChatML 模板与停止符）见 [操作指南.md](操作指南.md) 第 8 节。

## 5. 脚本说明

### 业务脚本

| 脚本 | 作用 | 关键输入 | 关键输出 |
| :--- | :--- | :--- | :--- |
| `clean_data.py` | 校验/过滤/去重，规范化成可训练 jsonl | `--input`（默认 `LCCC-base-close\LCCC-base_valid.jsonl`） | `--output`（默认 `dataset\LCCC-base_valid_clean.jsonl`） |
| `finetune_qwen.py` | 加载 4-bit 基座 → 注入 LoRA → 读清洗数据 → 训练并保存适配器 | `--data`/`--max_steps` 等 9 个参数 | `lora_model\`、`outputs\` |
| `merge_16bit_peft.py` | `PeftModel.merge_and_unload()` 把 LoRA 合进 16-bit 基座 | `models\Qwen3-4B-Instruct-16bit\` + `lora_model\` | `merged_16bit_model\` |
| `export_gguf.py` | 备选：Unsloth 直接导出 GGUF（NF4 基座会报错，当前未采用） | `--lora` | `--output`（默认 `gguf_output`） |
| `model_test.py` | 基座 + 适配器加载后推理一句，验证微调效果 | `lora_model\` | 终端输出 |
| `download_model.py` | 早期下载脚本（现直接用 `hf download`） | 仓库 ID | `models\Qwen3-4B-Instruct-bnb-4bit\` |

### `finetune_qwen.py` 命令行参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--data` | `dataset\LCCC-base_train_clean.jsonl` | 训练数据（清洗后训练集，671 万条） |
| `--model` | `models\Qwen3-4B-Instruct-bnb-4bit` | 本地 4-bit 基座 |
| `--output` | `...\outputs` | 训练日志与检查点 |
| `--lora` | `...\lora_model` | LoRA 适配器输出 |
| `--hf_cache` | `D:\00_Inbox\huggingface_cache` | HF 缓存 |
| `--max_steps` | `2000` | 训练步数（671 万条下约 0.24 epoch） |
| `--batch_size` | `2` | 批大小 |
| `--grad_accum` | `4` | 梯度累积 |
| `--lr` | `2e-4` | 学习率 |

例：用训练集跑 1000 步

```cmd
python code\finetune_qwen.py --data "D:\01_Project\project\2026-project18\dataset\LCCC-base-close\LCCC-base_train.jsonl" --max_steps 1000
```

### `clean_data.py` 命令行参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--input` | `dataset\LCCC-base-close\LCCC-base_valid.jsonl` | 原始 jsonl |
| `--output` | `dataset\LCCC-base_valid_clean.jsonl` | 清洗后 jsonl（训练集需显式传 `--input/--output`） |
| `--max_chars` | `4096` | 单样本字符数上限 |
| `--no_dedup` | 关闭 | 不去重 |
| `--dryrun` | 关闭 | 只统计，不写文件 |

清洗规则共 9 条（JSON 错误、缺 `conversations`、非法 role、空 content、`<tool_call>`、ChatML 标签、超长、非 user 开头、重复），逐条说明见 [操作指南.md](操作指南.md) 第 3.2 节。

### 自检脚本（非流程必需，可随时单独运行）

| 脚本 | 检查内容 |
| :--- | :--- |
| `test_torch.py` | torch 版本、CUDA 版本、是否可用、计算能力、GPU 矩阵运算 |
| `test_unsloth.py` | Unsloth 能否导入、CUDA 是否正常 |
| `test_triton.py` | Triton 能否导入 |
| `test_3.py` | 以上三项合并，一次跑完 |
| `test_dataset.py` | 打印数据首行的 JSON 结构，验证 `conversations` 字段 |

## 6. 数据说明

- 数据来源：LCCC-base（close 版）中文闲聊，已转换为 `conversations` 多轮对话格式（role 只有 user/assistant）。
- 规模：train 682 万条（1326.2 MB）、valid 2 万条（3.9 MB）、test 1 万条（2.0 MB）。
- 清洗后产物（都放在 `dataset\` 根目录）：训练集 `LCCC-base_train_clean.jsonl`，6716583 条 / 1298.9 MB（`finetune_qwen.py` 的默认训练集）；验证集 `LCCC-base_valid_clean.jsonl`，19986 条 / 3.84 MB（可做快速链路验证或留作评估）。
- **`<tool_call>` 与数据无关**：实测三份数据的 `<tool_call>`、`<|im_start|>`、非法 role 命中数全为 0；模型输出 `<tool_call>` 来自基座在 ChatML 非 thinking 模式下的固有倾向。对策是手动拼接 ChatML（见 [操作指南.md](操作指南.md) 第 3.3 节）+ Ollama 侧的停止符。
- 数据被 `.gitignore` 忽略，换机后需从原项目 `2026-project17` 或数据备份恢复。
- `dataset\LCCC-base_test.jsonl` 与 `dataset\LCCC-base-close\LCCC-base_test.jsonl` 内容相同，前者可删。

## 7. 关键参数

| 项目 | 取值 | 说明 |
| :--- | :--- | :--- |
| 训练基座 | Qwen3-4B-Instruct-bnb-4bit | QLoRA 4-bit 加载，8GB 显存的关键 |
| 合并基座 | Qwen3-4B-Instruct-16bit | bf16 载入，用于 `merge_and_unload` |
| 微调方式 | LoRA，`r=16`，`lora_alpha=16`，`lora_dropout=0` | 目标模块 `q/k/v/o_proj` + `gate/up/down_proj` |
| 最大序列长度 | 2048 | `--max_chars 4096` 与之匹配 |
| 批大小 / 梯度累积 | 2 / 4（等效 8） | 显存不足时改 1 / 8 |
| 学习率 / 步数 | `2e-4` / 500 | 线性衰减，warmup 5 步 |
| 优化器 / 精度 | `adamw_8bit` / bf16 | `weight_decay=0.01`，seed 3407 |
| 对话模板 | 手动 ChatML | `<|im_start|>role\ncontent<|im_end|>\n` |
| GGUF 精度 | f16 → Q4_K_M | 先转 f16，再量化到约 2.5GB |
| 推理采样 | temperature 0.7 / top_p 0.9 / repeat_penalty 1.1 | 见 Modelfile |
| 语料规模建议 | 3 万~5 万条（清洗后） | 当前 19986 条偏少，loss 收敛有限 |

## 8. 常见问题

| 现象 | 处理 |
| :--- | :--- |
| 训练 loss 不降 | 数据换成清洗后的 valid/train 集，步数加到 500~1000，学习率可保持 2e-4 |
| 模型输出 `<tool_call>` | 数据侧已排除该原因：用手动 ChatML 模板训练 + Modelfile 加 `<tool_call>` 停止符 |
| 合并时报 torchao 错误 | `pip install torchao==0.15.0` + `pip install "peft<0.16"`；合并完把 `peft` 装回 `>=0.18.0` |
| 合并时显存不足 | 改 `torch.float16`，或加 `load_in_8bit=True`；`device_map="auto"` 会自动卸载到内存 |
| 转换 GGUF 报 RoPE 警告 | 无害，忽略 |
| 量化时找不到 `llama-quantize.exe` | 下载 llama.cpp 预编译版（Windows x64 CPU）解压到 `2026-project17\llama-cpp-bin` |
| Ollama 读 `/mnt/d` 权限错误 | 先把 GGUF 复制到 WSL 家目录再 `ollama create` |
| `CUDA out of memory` | `--batch_size 1 --grad_accum 8`；仍不够则把 `max_seq_length` 降到 1024 |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch，用 cu128 版本重装 |
| `ModuleNotFoundError: No module named 'triton'` | 装 `triton-windows`（`pip install unsloth` 会带上），确认在 `unsloth311` 环境 |
| 下载中断 | `set HF_ENDPOINT=https://hf-mirror.com` 后重跑同一命令，支持续传 |
| 找不到 `lora_model` / 产物落错位置 | 含相对路径的老脚本需在项目根目录运行；新脚本均为绝对路径 |

## 9. 注意事项

- **重训会作废下游产物**：`lora_model/` 一变，`merged_16bit_model/`、`gguf_output/` 就要重跑第 5~7 步。
- **工作目录**：多数脚本用绝对路径，但统一在项目根目录执行最省心；`llama.cpp` 的转换/量化命令必须 `cd` 到工具目录。
- **两个 HF 缓存口径**：`finetune_qwen.py` 用 `D:\00_Inbox\huggingface_cache`，`merge_16bit_peft.py` 不设置（用默认缓存）——因为都是本地路径，不影响结果，但改缓存位置时要留意。
- **sm_120 显卡对版本敏感**：torch 必须是 cu128 构建，xformers 必须与之匹配；合并阶段还要临时降级 `torchao`/`peft`。
- **大文件不入库**：模型、数据、合并产物、GGUF、LoRA 适配器都被 `.gitignore` 忽略，克隆仓库只得到代码和文档。
- **磁盘占用**（实测）：`models\` 约 14.2 GB、`merged_16bit_model\` 约 7.5 GB、`gguf_output\` 约 7.5 GB、`dataset\` 约 1.3 GB、`lora_model\` 约 137 MB、`outputs\` 约 200 MB，合计 **约 31 GB**。
  - 生成 f16 GGUF 后，`merged_16bit_model\` 即为可删的中间产物（省 7.5 GB）；需要重跑转换时再从 LoRA 合并一次即可。
  - 若已量化出 Q4_K_M，f16 GGUF 也可在确认无误后删除（再省 7.5 GB）。
