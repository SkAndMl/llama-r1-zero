from datasets import load_dataset, concatenate_datasets
import re
import json

ds = load_dataset("allenai/sciq")
train_ds = concatenate_datasets([ds['train'], ds['validation']])
test_ds = ds['test']

train_dicts = []
test_dicts = []


for row in train_ds:
    q = row['question']
    a = row['correct_answer']
    if a is not None:
        train_dicts.append({
            "question": q,
            "answer": a
        })

for row in test_ds:
    q = row['question']
    a = row['correct_answer']
    if a is not None:
        test_dicts.append({
            "question": q,
            "answer": a
        })

print(f'num train: {len(train_dicts)}')
print(f'num test: {len(test_dicts)}')

with open("data/sciq_train.jsonl", "w") as f:
    for row in train_dicts:
        json.dump(row, f)
        f.write("\n")

with open("data/sciq_test.jsonl", "w") as f:
    for row in test_dicts:
        json.dump(row, f)
        f.write("\n")