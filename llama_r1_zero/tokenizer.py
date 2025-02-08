from typing import Dict, List, cast, Sequence, Literal, TypedDict
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
import torch

Role = Literal['system', 'user', 'assistant']

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]


class Tokenizer:
 
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>", 
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self,
        s: str,
        bos: bool,
        eos: bool,
        allowed_special=set(),
        disallowed_special=()
    ) -> List[int]:

        t: List[int] = self.model.encode(s,
                                         allowed_special=allowed_special,
                                         disallowed_special=disallowed_special) 
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def encode_batch(
            self,
            seqs: List[str], 
            bos: bool, 
            eos: bool,
            padding_side: str ='right', 
            allowed_special=set(), 
            disallowed_special=()
    ) -> torch.Tensor:
        
        max_len = 0
        tokens = []
        for s in seqs: 
            _toks = self.encode(s, bos, eos, allowed_special, disallowed_special)
            tokens.append(_toks)
            max_len = max(max_len, len(_toks))
        
        tokens_tensor = torch.full((len(seqs), max_len), fill_value=self.pad_id, dtype=torch.long)
        for i, toks in enumerate(tokens):
            if padding_side=='right':
                tokens_tensor[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            else:
                tokens_tensor[i, -len(toks):] = torch.tensor(toks, dtype=torch.long)
        
        return tokens_tensor

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(List[int], t))
    
    def decode_batch(self, texts: Sequence[Sequence[int]]) -> str:
        return [self.decode(t) for t in texts]
        
    def encode_header(self, message):
        tokens = []
        tokens.append(self.special_tokens["<|start_header_id|>"])
        tokens.extend(self.encode(message["role"], bos=False, eos=False))
        tokens.append(self.special_tokens["<|end_header_id|>"])
        tokens.extend(self.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens