r"""数据清洗/校验脚本：把原始 LCCC jsonl 规范化为可训练格式。

用途
----
`finetune_qwen.py` 默认读取 `dataset\LCCC-base_valid_clean.jsonl`。
本脚本负责生成这个文件：逐行校验并过滤，再规范化写出（UTF-8、无 BOM）。

清洗规则（命中任一即丢弃该样本，并计入统计）
--------------------------------------------
1. JSON 解析失败
2. 缺少 `conversations` 字段，或它不是列表 / 为空
3. 出现非 user/assistant/system 的 role，或缺少 content
4. content 为空或纯空白
5. 含 `<tool_call>` / `</tool_call>` 标记
6. 含 `<|im_start|>` / `<|im_end|>` 标记（避免与训练模板重复嵌套）
7. 总字符数超过 `--max_chars`（默认 4096，配合 max_seq_length=2048 使用）
8. 对话不是以 user 开始（规范化：从第一个 user 截断；截断后为空则丢弃）
9. 重复样本（按规范化后的整段文本去重）

用法
----
    python code\clean_data.py                       # 默认清洗 valid 集
    python code\clean_data.py --input  <原始.jsonl> --output <清洗后.jsonl>
    python code\clean_data.py --input ... --output ... --max_chars 8192 --no_dedup
    python code\clean_data.py --dryrun               # 只统计不写文件

说明
----
- 只读输入、只写输出文件，不会改动原始数据。
- 本项目的 LCCC 数据经实测本身无 `<tool_call>` 污染；该规则用于兜底，
  因为 `<tool_call>` 更多来自基座模型在 ChatML 下的行为，而非数据本身。
"""

import argparse
import io
import json
import os
from collections import Counter

DEFAULT_INPUT = r"D:\01_Project\project\2026-project18\dataset\LCCC-base-close\LCCC-base_valid.jsonl"
DEFAULT_OUTPUT = r"D:\01_Project\project\2026-project18\dataset\LCCC-base_valid_clean.jsonl"

VALID_ROLES = ("user", "assistant", "system")
BAD_MARKERS = ("<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>")


def normalize(conversations):
    """规范化对话：从第一个 user 开始，返回 (新对话, 是否发生截断)。"""
    start = 0
    for i, msg in enumerate(conversations):
        if msg.get("role") == "user":
            start = i
            break
    else:
        return None, False
    trimmed = conversations[start:]
    return trimmed, start > 0


def check(conversations, max_chars):
    """返回 None 表示通过；否则返回丢弃原因。"""
    if not conversations:
        return "empty_conversations"
    total = 0
    for msg in conversations:
        if not isinstance(msg, dict):
            return "message_not_object"
        if msg.get("role") not in VALID_ROLES:
            return "invalid_role"
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return "empty_content"
        total += len(content)
        for marker in BAD_MARKERS:
            if marker in content:
                return "bad_marker:" + marker
    if total > max_chars:
        return "too_long"
    return None


def main():
    parser = argparse.ArgumentParser(description="清洗/校验训练数据 jsonl")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="原始 jsonl 路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="清洗后 jsonl 路径")
    parser.add_argument("--max_chars", type=int, default=4096, help="单样本字符数上限")
    parser.add_argument("--no_dedup", action="store_true", help="关闭按整段文本去重")
    parser.add_argument("--dryrun", action="store_true", help="只统计，不写输出文件")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    reason_counter = Counter()
    seen = set()
    kept = 0
    total = 0
    turns_counter = Counter()
    total_chars = 0

    out_file = None
    if not args.dryrun:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out_file = io.open(args.output, "w", encoding="utf-8", newline="\n")

    try:
        with io.open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    reason_counter["json_error"] += 1
                    continue

                conversations = record.get("conversations")
                if not isinstance(conversations, list):
                    reason_counter["missing_or_bad_conversations"] += 1
                    continue

                conversations, trimmed = normalize(conversations)
                if not conversations:
                    reason_counter["not_starting_with_user"] += 1
                    continue

                reason = check(conversations, args.max_chars)
                if reason is not None:
                    reason_counter[reason] += 1
                    continue

                if not args.no_dedup:
                    key = json.dumps(conversations, ensure_ascii=False, sort_keys=True)
                    if key in seen:
                        reason_counter["duplicate"] += 1
                        continue
                    seen.add(key)

                kept += 1
                turns_counter[len(conversations)] += 1
                total_chars += sum(len(m["content"]) for m in conversations)
                if trimmed:
                    reason_counter["trimmed_to_first_user"] += 1

                if out_file is not None:
                    out_file.write(json.dumps({"conversations": conversations},
                                              ensure_ascii=False) + "\n")
    finally:
        if out_file is not None:
            out_file.close()

    # 报告
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出: {'(dryrun，未写文件)' if args.dryrun else args.output}")
    print("=" * 60)
    print(f"原始样本数 : {total}")
    print(f"保留样本数 : {kept}  ({kept / max(total, 1) * 100:.2f}%)")
    print(f"丢弃样本数 : {total - kept}")
    if kept:
        print(f"平均字符数 : {total_chars / kept:.1f}")
        print(f"轮数分布   : {turns_counter.most_common(8)}")
    if reason_counter:
        print("丢弃原因统计:")
        for reason, count in reason_counter.most_common():
            print(f"  {reason:32s} {count}")
    if not args.dryrun:
        print(f"输出文件大小: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
