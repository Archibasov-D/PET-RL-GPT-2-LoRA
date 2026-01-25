import torch
class IMDBPairwiseDataset(torch.utils.data.Dataset):
    """ A dataset of all possible pairs of chosen and texts in TRL reward training format """
    def __init__(self, imdb, tokenizer, accepted_label: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = [row['text'] for row in imdb if row['label'] == accepted_label]
        self.rejected_texts = [row['text'] for row in imdb if row['label'] != accepted_label]
        assert self.chosen_texts, f"no texts with label {accepted_label}"
        print(f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs")

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)  # all pairs

    def __getitem__(self, index: int):
        chosen = self.tokenizer(self.chosen_texts[index // len(self.chosen_texts)], truncation=True)
        rejected = self.tokenizer(self.rejected_texts[index % len(self.chosen_texts)], truncation=True)
        return dict(chosen_input_ids=chosen['input_ids'], chosen_attention_mask=chosen['attention_mask'],
                   rejected_input_ids=rejected['input_ids'], rejected_attention_mask=rejected['attention_mask'])
        

class IMDBPairwiseDataset_test(torch.utils.data.Dataset):
    """ A dataset of all possible pairs of chosen and texts in TRL reward training format """
    def __init__(self, imdb, accepted_label: int):
        super().__init__()
        self.chosen_texts = [row['text'] for row in imdb if row['label'] == accepted_label]
        self.rejected_texts = [row['text'] for row in imdb if row['label'] != accepted_label]
        assert self.chosen_texts, f"no texts with label {accepted_label}"
        print(f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs")

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)  # all pairs

    def __getitem__(self, index: int):
        return {'chosen' : self.chosen_texts[index // len(self.chosen_texts)],
                    'rejected':self.rejected_texts[index % len(self.chosen_texts)]}


def patched_forward(self, original_forward, *args, **kwargs):
    """Удаляем неподдерживаемые параметры"""
    kwargs.pop("use_cache", None)
    kwargs.pop("cache_position", None)
    return original_forward(*args, **kwargs)
