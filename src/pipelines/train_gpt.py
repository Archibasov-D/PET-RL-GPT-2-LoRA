# Напишем зачатки ревью:
import random
import torch  
from transformers import AutoTokenizer

class LengthSampler:
    """Простая замена для trl.LengthSampler"""
    def __init__(self, min_value: int, max_value: int):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self) -> int:
        return random.randint(self.min_value, self.max_value)


def select_query_and_tokenize(sample, main_tokenizer, sample_length):
    
    query_ids = main_tokenizer.encode(sample["text"])[: sample_length()]
    sample["prompt"] = main_tokenizer.decode(query_ids)  # query is the only required column
    sample["input_ids"] = query_ids  # to avoid re-tokenizing later
    return sample  

def compute_reward(reward_model, reward_tokenizer, device, prompts=None, completions=['default'], completion_ids=None, **kwargs):
  inputs = reward_tokenizer(completions, truncation=True, padding=True, return_tensors='pt').to(device)
  with torch.no_grad():
    return reward_model(**inputs).logits[:, 0].to('cpu').tolist()
