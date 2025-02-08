from dotenv import load_dotenv, find_dotenv
from .model import Transformer, ModelArgs
from .tokenizer import Tokenizer, Dialog
import os
import json
import torch
from torch.nn import functional as F
from typing import List

load_dotenv(find_dotenv())


class Llama:

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
    
    @staticmethod
    def build(
        ckpt_dir: str, 
        max_batch_size: int,  
        max_seq_len: int,
        device: str = "cpu",
    ):
        
        tokenizer = Tokenizer(model_path=os.path.join(ckpt_dir, "tokenizer.model"))
        with open(os.path.join(ckpt_dir, 'params.json'), 'r') as f:
            params = json.load(f)
        args = ModelArgs(
            max_batch_size = max_batch_size,
            max_seq_len = max_seq_len,
            device = device,
            **params
        )
        assert args.vocab_size == tokenizer.n_words

        wts = torch.load(os.path.join(ckpt_dir, "consolidated.00.pth"), map_location=device)
        model = Transformer(args)
        model.load_state_dict(wts)

        return Llama(model, tokenizer, args)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        temperature: float=0.6,
        top_p: float = 0.8,
        num_generations: int = 1,
        max_gen_len: int = 1024,
        echo: bool = False,
        return_tensors: bool = False
    ) -> List[List[int]]:
        
        assert len(prompt_tokens)*num_generations<=self.args.max_batch_size

        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(self.args.max_seq_len, max_gen_len+max_prompt_len)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))
        bsz = len(prompt_tokens)
        input_tokens = torch.full((bsz, max_prompt_len), 
                                  fill_value=self.tokenizer.special_tokens['<|end_of_text|>'], 
                                  dtype=torch.long, 
                                  device=self.args.device)
        for i, prompt_token in enumerate(prompt_tokens):
            input_tokens[i, -len(prompt_token):] = torch.tensor(prompt_token,
                                                                dtype=torch.long,
                                                                device=self.args.device)
        input_tokens = torch.repeat_interleave(input_tokens,
                                               repeats=num_generations,
                                               dim=0)
        bsz *= num_generations
        eos_reached = torch.tensor([False]*bsz, device=self.args.device)
        prev_pos = 0
        for cur_pos in range(max_prompt_len, total_len):
            logits = self.model.forward(
                x=input_tokens[:, prev_pos:cur_pos],
                start_pos=prev_pos
            ) # bsz, l+1, v
            if temperature>0.0:
                probs = F.softmax(logits[:, -1]/temperature,
                                  dim=-1)
                next_token = sample_top_p(probs, p=top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1) # bsz, 1
            
            input_tokens = torch.cat(
                tensors=[input_tokens, next_token],
                dim=-1
            )

            eos_reached |= torch.isin(next_token.reshape(-1), stop_tokens)
            if eos_reached.all(): break

            prev_pos = cur_pos
        
        if return_tensors: return input_tokens

        outs = []
        for i, completion in enumerate(input_tokens.tolist()):

            start = max_prompt_len-len(prompt_tokens[i//num_generations]) if echo else max_prompt_len
            toks: List[int] = completion[start:]
            if self.tokenizer.special_tokens['<|end_of_text|>'] in toks:
                idx = toks.rindex(self.tokenizer.special_tokens['<|end_of_text>'])
                toks = toks[:idx]
            outs.append(toks)
        
        return outs


    def text_completion(
        self,
        prompts: List[Dialog],
        temperature: float=0.6,
        top_p: float=0.8,
        num_generations: int=1,
        max_gen_len: int=-1,
        echo: bool = False 
    ) -> List[str]:
        
        if max_gen_len == -1:
            max_gen_len = self.args.max_seq_len-1
        
        prompt_tokens = [
            self.tokenizer.encode_dialog_prompt(dialog=prompt) 
            for prompt in prompts
        ]
        generated_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            num_generations=num_generations,
            max_gen_len=max_gen_len,
            echo=echo
        )

        return [
            self.tokenizer.decode(g) for g in generated_tokens
        ]



def sample_top_p(probs, p) -> torch.Tensor:
    
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum-probs_sort>p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)
    return next_token