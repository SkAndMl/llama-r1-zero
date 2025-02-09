# reference:https://github.com/huggingface/trl/blob/949db2357e62d2f0a34decfc5e87eeeea0c6d72c/trl/trainer/grpo_trainer.py#L274-L281 

import torch
from typing import List, Optional, Union, Callable
from .llama import Llama
from .tokenizer import Tokenizer

class GRPOLoss:
    def __init__(
        self,
        tokenizer: Tokenizer,
        system_prompt: str,
        reward_funcs: List[Callable],
        ref_model: Optional[Llama] = None,
        beta: float = 0.01,
        num_generations: int = 4,
        max_new_tokens: int = None,
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.reward_funcs = reward_funcs
        self.ref_model = ref_model
        self.beta = beta
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else 1024
        self.metrics = {"reward": [], "reward_std": [], "kl": []}

    def get_per_token_logprobs(self, model: Llama, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.clone()
        logits = model.model.forward(input_ids, start_pos=0)  # (bs, seq_len, vocab_size)
        logits = logits[:, :-1, :]  # exclude last prediction
        input_ids = input_ids[:, 1:]  # exclude first input ID
        
        per_token_logprobs = []
        for logits_row, ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_logprob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logprobs.append(token_logprob)
        
        return torch.stack(per_token_logprobs)

    def compute_rewards(self, completions: List[List[str]], ground_truths: List[str]) -> torch.Tensor:
        # completions -> bs, g
        # ground_truths -> bs
        bs = len(completions)
        # for i, completion in enumerate(completions):
        #     print(f'completion: {i+1}\n{completion}')
        #     print('-'*50)
        rewards = torch.zeros((bs//self.num_generations, self.num_generations))
        for i in range(0, len(completions), self.num_generations):
            for reward_func in self.reward_funcs:
                _rewards = reward_func(
                    completions=completions[i:i+self.num_generations],
                    ground_truths=[ground_truths[i//self.num_generations]]*self.num_generations
                )
                rewards[i//self.num_generations, :] += torch.tensor(_rewards).float()

        return rewards.view(-1,) # bs*g

    def compute_loss(self, model: Llama, prompts: List[str], ground_truths: List[str]) -> Union[torch.Tensor, tuple]:
        
        prompt_inputs = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": p}
            ]
            for p in prompts
        ]
        prompts_tokens = [self.tokenizer.encode_dialog_prompt(p) for p in prompt_inputs]
        max_prompt_len = max(len(p) for p in prompts_tokens)
    
        with torch.inference_mode():
            generated_ids: torch.Tensor = model.generate(
                prompt_tokens=prompts_tokens,
                temperature=0.6,
                top_p=0.8,
                num_generations=self.num_generations,
                max_gen_len=self.max_new_tokens,
                return_tensors=True
            )
            

        completion_ids = generated_ids[:, max_prompt_len:]

        per_token_logps = self.get_per_token_logprobs(model, generated_ids)
        per_token_logps = per_token_logps[:, max_prompt_len-1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self.get_per_token_logprobs(self.ref_model, generated_ids)
            else:
                ref_per_token_logps = self.get_per_token_logprobs(model, generated_ids)
            ref_per_token_logps = ref_per_token_logps[:, max_prompt_len-1:]

        is_eos = completion_ids == self.tokenizer.eos_id
        device = model.args.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completions = []
        for _c in completion_ids.tolist():
            idx = len(_c)
            if self.tokenizer.special_tokens['<|eot_id|>'] in _c:
                idx = _c.index(self.tokenizer.special_tokens['<|eot_id|>'])
            completions.append(_c[:idx])

        completions = self.tokenizer.decode_batch(completions)
        
        rewards = self.compute_rewards(completions, ground_truths)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                      (ref_per_token_logps - per_token_logps) - 1
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        self.metrics["reward"].append(rewards.mean().item())
        self.metrics["reward_std"].append(std_grouped_rewards.mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self.metrics["kl"].append(mean_kl.item())

        return loss