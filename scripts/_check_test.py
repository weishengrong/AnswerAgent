import json

data = json.load(open("scripts/intent_test_cases.json", "r", encoding="utf-8"))
print(f"测试集条数: {len(data)}")

cats = {}
for d in data:
    c = d["category"]
    cats[c] = cats.get(c, 0) + 1

print(f"类别数: {len(cats)}")
for c in sorted(cats):
    print(f"  {c}: {cats[c]}条")

intents = {}
for d in data:
    i = d["expected_intent"]
    intents[i] = intents.get(i, 0) + 1
print(f"\n意图分布:")
for i in sorted(intents):
    print(f"  {i}: {intents[i]}条")
