from datasets import load_dataset
import re
import json

ds = load_dataset("openai/gsm8k", 'main')
train_ds = ds['train']
test_ds = ds['test']

answer_pattern = r'#{4}\s(\d+)'

train_dicts = []
test_dicts = []


for row in train_ds:
    q = row['question']
    a = row['answer']
    match = re.search(answer_pattern, a)
    if match:
        train_dicts.append({
            "question": q,
            "answer": match.group(1)
        })

for row in test_ds:
    q = row['question']
    a = row['answer']
    match = re.search(answer_pattern, a)
    if match:
        test_dicts.append({
            "question": q,
            "answer": match.group(1)
        })

print(f'num train: {len(train_dicts)}')
print(f'num test: {len(test_dicts)}')

with open("data/gsm8k_train.jsonl", "w") as f:
    for row in train_dicts:
        json.dump(row, f)
        f.write("\n")

with open("data/gsm8k_test.jsonl", "w") as f:
    for row in test_dicts:
        json.dump(row, f)
        f.write("\n")