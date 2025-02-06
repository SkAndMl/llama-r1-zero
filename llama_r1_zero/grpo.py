# reference:https://github.com/huggingface/trl/blob/949db2357e62d2f0a34decfc5e87eeeea0c6d72c/trl/trainer/grpo_trainer.py#L274-L281 

import torch
from typing import List, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer

class GRPOLoss:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_funcs: List[Callable],
        ref_model: Optional[PreTrainedModel] = None,
        beta: float = 0.01,
        max_prompt_length: Optional[int] = None,
        num_generations: int = 4,
        max_new_tokens: int = None
    ):
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.ref_model = ref_model
        self.beta = beta
        self.max_prompt_length = max_prompt_length
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else 256
        self.metrics = {"reward": [], "reward_std": [], "kl": []}

    def get_per_token_logprobs(self, model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:

        logits = model(input_ids).logits  # (bs, seq_len, vocab_size)
        logits = logits[:, :-1, :]  # exclude last prediction
        input_ids = input_ids[:, 1:]  # exclude first input ID
        
        per_token_logprobs = []
        for logits_row, ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_logprob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logprobs.append(token_logprob)
        
        return torch.stack(per_token_logprobs)

    def compute_rewards(self, completions: List[List[str]]) -> torch.Tensor:
        # completions -> bs, g
        bs = len(completions)
        rewards = torch.zeros((bs, self.num_generations))
        for i, group in enumerate(completions):
            for reward_func in self.reward_funcs:
                _rewards = reward_func(completions=group)
                rewards[i, :] += torch.tensor(_rewards).float()

        return rewards.view(-1,) # bs*g

    def compute_loss(self, model: PreTrainedModel, prompts: List[str]) -> Union[torch.Tensor, tuple]:
            
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        ).to(model.device)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        with torch.inference_mode():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample = True,
                top_p = 0.9,
                temperature = 0.6,
                num_return_sequences=self.num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = generated_ids[:, prompt_length:]

        # Get log probs for completions
        per_token_logps = self.get_per_token_logprobs(model, generated_ids)
        per_token_logps = per_token_logps[:, prompt_length-1:]

        # Get reference model log probs
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self.get_per_token_logprobs(self.ref_model, generated_ids)
            else:
                ref_per_token_logps = self.get_per_token_logprobs(model, generated_ids)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]

        # Create completion mask
        is_eos = completion_ids == self.tokenizer.eos_token_id
        device = model.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode completions for reward computation
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        rewards = self.compute_rewards(completions)

        # Compute grouped rewards statistics
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                      (ref_per_token_logps - per_token_logps) - 1

        # Compute final loss
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        self.metrics["reward"].append(rewards.mean().item())
        self.metrics["reward_std"].append(std_grouped_rewards.mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self.metrics["kl"].append(mean_kl.item())

        return loss