from llama_r1_zero.grpo import GRPOLoss
from llama_r1_zero.llama import Llama
from llama_r1_zero.prompts import SYSTEM_PROMPT
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
from dotenv import load_dotenv, find_dotenv
import torch
from torch.optim import SGD
from llama_r1_zero.rewards import format_reward, accuracy_reward, complexity_reward, similarity_reward

load_dotenv(find_dotenv())


class ReasoningData(Dataset):

    def __init__(self, json_paths: List[str]) -> None:

        self.data = []
        for path in json_paths:
            with open(path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str]:
        return self.data[idx]['question'], self.data[idx]['answer']


train_json_paths = ['data/sciq_train.jsonl', 'data/gsm8k_train.jsonl']
test_json_paths = ['data/sciq_test.jsonl', 'data/gsm8k_test.jsonl']

batch_size = 8
num_generations = 4
print_generation_every = 10
eval_every = 100
print_loss_every = 50
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = ReasoningData(train_json_paths)
test_ds = ReasoningData(test_json_paths)

train_dl = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True
)

test_dl = DataLoader(
    dataset=test_ds,
    batch_size=batch_size
)

llama: Llama = Llama.build(
    ckpt_dir=os.environ.get("MODEL_PATH"),
    max_batch_size=batch_size*num_generations,
    device=device
)

loss_fn = GRPOLoss(
    tokenizer=llama.tokenizer,
    system_prompt=SYSTEM_PROMPT,
    reward_funcs=[accuracy_reward, complexity_reward, format_reward, similarity_reward]
    # should remove complexity reward for faster training
    # added similarity reward, for backprop during the initial iterations
)
# let's use sgd for the initial training process
# to have lesser params on the gpu and for faster training
optimizer = SGD(
    params = llama.model.parameters(),
    lr = learning_rate
)


@torch.inference_mode()
def eval_model():

    llama.model.eval()
    avg_loss = 0
    for i, (question, answer) in enumerate(test_dl):
        
        loss = loss_fn.compute_loss(
            model=llama,
            prompts=question,
            ground_truths=answer,
            print_generations=(i+1)%print_generation_every==0
        )
        avg_loss += loss.item()

        if (i+1)%print_loss_every==0:
            print(f'eval loss: {avg_loss/(i+1)}')
    
    print(f'avg eval loss: {avg_loss/len(test_dl)}')
    llama.model.train()


train_loss = 0
for i, (question, answer) in enumerate(train_dl):

    loss = loss_fn.compute_loss(
        model=llama,
        prompts=question,
        ground_truths=answer,
        print_generations=(i+1)%print_generation_every==0
    )
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%print_loss_every==0:
        print(f'train loss: {train_loss/(i+1)}')


    if (i+1)%eval_every==0 or i==len(train_dl)-1:
        eval_model()


print(f'avg train loss: {train_loss/len(train_dl)}')