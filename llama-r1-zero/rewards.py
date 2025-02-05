import re
from typing import List

def format_reward(completions, **kwargs) -> List[float]:
    """
    reward of 1 the completion follows the pattern:
    <think>...</think><answer>...</answer>
    based on: https://huggingface.co/docs/trl/main/en/grpo_trainer
    """
    pattern = r'^<think>.*</think><answer>.*</answer>$'
    matches = [re.match(pattern, completion) 
               for completion in completions]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward(completions, ground_truths, **kwargs) -> List[float]:
    """
    reward of 1 if the answer matches the ground truth
    """
    pattern = r'\\boxed\{(.*?)\}'
    matches = [re.search(pattern, completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c==gt else 0.0 for (c, gt) in zip(contents, ground_truths)]


def complexity_reward(completions, **kwargs) -> List[float]:
    """
    Reward is proportional to the length of the reasoning tokens inside <think>...</think>.
    """
    pattern = r'<think>(.*?)</think>'
    reasoning_lengths = [
        len(re.search(pattern, completion).group(1).split()) if re.search(pattern, completion) else 0
        for completion in completions
    ]

    max_length = max(reasoning_lengths) + 1e-5  # avoid division by zero
    return [length / max_length for length in reasoning_lengths]