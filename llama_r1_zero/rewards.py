import re
from typing import List
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    pattern = r'<answer>(.*?)</answer>'
    matches = [re.search(pattern, completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c==gt else 0.0 for (c, gt) in zip(contents, ground_truths)]


def complexity_reward(completions, ground_truths, **kwargs) -> List[float]:
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

def similarity_reward(completions, ground_truths, **kwargs) -> List[float]:
    """
    Reward based on cosine similarity between the generated <answer> and the ground truth.
    Uses sentence embeddings for semantic similarity.
    """
    pattern = r'<answer>(.*?)</answer>'
    matches = [re.search(pattern, completion) for completion in completions]
    answers = [match.group(1) if match else completions[i][-20:] for i,match in enumerate(matches)]

    embeddings_completions = embedder.encode(answers, convert_to_tensor=True)
    embeddings_ground_truths = embedder.encode(ground_truths, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(embeddings_completions, embeddings_ground_truths).diagonal()

    return similarities.tolist()