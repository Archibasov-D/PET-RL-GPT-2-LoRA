import torch
import trl
import transformers
from transformers import AutoModelForSequenceClassification
import datasets
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
from trl import GRPOTrainer, GRPOConfig, RewardTrainer, RewardConfig
from src.pipelines.train_reward_model import IMDBPairwiseDataset, IMDBPairwiseDataset_test, patched_forward
from box import ConfigBox
from src.utils.decorator import parser
import os
from dotenv import load_ext


@parser(prog_name="Train reward model", dscr="Download model and dataset| Create dataset for train | Apply RewardTrainer")
def train_reward_model(params):
    np.random.seed(params.base.random_seed)
    random.seed(params.base.random_seed)
    torch.manual_seed(params.base.random_seed)
    torch.cuda.manual_seed(params.base.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imdb = load_from_disk(Path(params.download_hf_imdb.data_dir) / params.download_hf_imdb.hf_name)
    reward_data = IMDBPairwiseDataset_test(imdb,
                                       accepted_label=params.base.TARGET_LABEL)

    
    load_dotenv()
    token = os.getenv("HF_TOKEN") # загрузка токена через venv
    # конвертация в приемлимый формат для трейнера
    all_examples = []
    N = params.train_reward.size_of_train_reward_dataset
    pairs =  np.random.randint(0, len(reward_data) + 1, size=N)
    for i in tqdm(pairs):
        example = reward_data[i]


        if torch.is_tensor(example["chosen"]):
            example = {
                'chosen': example["chosen'"].tolist(),
                'rejected': example["rejected'"].tolist()
            }

        all_examples.append(example)


    hf_reward_data = Dataset.from_list(all_examples)
    # загрузка модели
    reward_model = AutoModelForSequenceClassification.from_pretrained(params.train_reward.reward_model_name,
                                                                      attn_implementation=params.train_reward.attn_implementation,
                                                                      device_map=device,
                                                                      num_labels=params.train_reward.num_labels)
    reward_model.config.use_cache = params.train_reward.reward_model_config_use_cache
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(params.train_reward.reward_model_name)

    # Применяем патч
    reward_model.forward = patched_forward.__get__(reward_model,
                                                   reward_model.__class__)
    # Используем библиотеку trl в ней есть трейнер аналогичный обычному от HF но для RL
    # Устанавливаем EOS
    if reward_tokenizer.eos_token is None:
        reward_tokenizer.eos_token = reward_tokenizer.sep_token
    # Устанавливаем PAD
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # Конфиг трейнера

    training_args = trl.RewardConfig(  # Аналог transformers.TrainingArguments
        output_dir= params.train_reward.output_dir,
        per_device_train_batch_size= params.train_reward.per_device_train_batch_size,
        gradient_accumulation_steps= params.train_reward.gradient_accumulation_steps,
        learning_rate= params.train_reward.learning_rate,
        max_steps= params.train_reward.max_steps,
        logging_steps= params.train_reward.logging_steps,
        max_length= params.train_reward.max_length,
        gradient_checkpointing= params.train_reward.gradient_checkpointing,
        bf16= torch.cuda.is_bf16_supported(),
        fp16= not torch.cuda.is_bf16_supported()    # fp16 для старых GPU
    )

    reward_tokenizer.model_max_length = params.train_reward.max_length

    trainer = trl.RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=hf_reward_data,
    peft_config=None,  # будем делать полный fine-tune
    push_to_hub=params.train_reward.push_to_hub, #Чтобы залить на HF НЕ ПРОВЕРЯЛ 
    hub_model_id=params.train_reward.hub_model_id, #Чтобы залить на HF НЕ ПРОВЕРЯЛ 
    processing_class = reward_tokenizer
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train_reward_model()

