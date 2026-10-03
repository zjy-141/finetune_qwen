import json

with open(r"D:\01_Project\project\2026-project18\dataset\LCCC-base_test.jsonl", "r", encoding="utf-8") as f:
    first = json.loads(f.readline())
    print(json.dumps(first, ensure_ascii=False, indent=2)[:500])
    print("...")