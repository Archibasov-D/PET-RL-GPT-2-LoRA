import peft
import torch
import wandb
import trl
import transformers
from transformers import AutoModelForSequenceClassification
import datasets
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
from src.pipelines.train_gpt import LengthSampler, select_query_and_tokenize, compute_reward
from trl import GRPOTrainer, GRPOConfig, RewardTrainer, RewardConfig
from box import ConfigBox
from src.utils.decorator import parser
import os
from dotenv import load_dotenv
from src.utils.folder_management import create_folders
from src.pipelines.train_reward_model import patched_forward
from functools import partial

@parser(prog_name="Train gpt model", dscr="Download reward_model and dataset| Create dataset for train gpt | Apply GRPOTrainer")
def train_gpt(params):
    np.random.seed(params.base.random_seed)
    random.seed(params.base.random_seed)
    torch.manual_seed(params.base.random_seed)
    torch.cuda.manual_seed(params.base.random_seed)
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()
    token_hf = os.getenv("HF_TOKEN") # загрузка токена через venv
    # загрузка готовой reward model
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(params.train_reward.reward_model_name)

    reward_model = AutoModelForSequenceClassification.from_pretrained(params.train_reward.hub_model_id,
                                                                      attn_implementation=params.train_reward.attn_implementation,
                                                                      device_map=device,
                                                                      num_labels=params.train_reward.num_labels)
    # Применяем патч - ? 
    reward_model.forward = patched_forward.__get__(reward_model,
                                                   reward_model.__class__)
    # Используем библиотеку trl в ней есть трейнер аналогичный обычному от HF но для RL
    # Устанавливаем EOS
    if reward_tokenizer.eos_token is None:
        reward_tokenizer.eos_token = reward_tokenizer.sep_token
    # Устанавливаем PAD
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token



    ##########################################################################



    main_tokenizer = transformers.AutoTokenizer.from_pretrained(params.train_gpt.main_model)
    main_model = transformers.AutoModelForCausalLM.from_pretrained(params.train_gpt.main_model, 
                                                                   attn_implementation=params.train_reward.attn_implementation,# НЕ ПРОВЕРЯЛ НО НУЖНО
                                                                   device_map= device, 
                                                                   use_safetensors= params.train_gpt.use_safetensors, 
                                                                   force_download= params.train_gpt.force_download, 
                                                                   #resume_download= params.train_gpt.resume_download НЕДАВНО УДАЛИЛИ ЭТОТ АРГУМЕНТ
                                                                   ) 
    
    imdb = load_from_disk(Path(params.download_hf_imdb.data_dir) / params.download_hf_imdb.hf_name)

    sample_length = LengthSampler(2, 8)
    imdb_for_rlhf = imdb.filter(lambda row: len(row['text']) > 200, batched=False)
    imdb_for_rlhf = imdb_for_rlhf.remove_columns(['label'])
    sample_length = LengthSampler(2, 8)  # use the first 2-8 tokens as query

    imdb_for_rlhf = imdb_for_rlhf.map(select_query_and_tokenize, 
                                      batched=False,
                                      fn_kwargs={"main_tokenizer": main_tokenizer, "sample_length": sample_length} )
    imdb_for_rlhf.set_format(type="torch")

    peft_config = peft.LoraConfig(
        task_type= peft.TaskType.CAUSAL_LM,
        r= params.train_gpt.lora_r,
        lora_alpha= params.train_gpt.lora_alpha,
        lora_dropout= params.train_gpt.lora_dropout,
        inference_mode= params.train_gpt.inference_mode,
    )
    

    main_tokenizer.pad_token = main_tokenizer.eos_token

    main_model = peft.get_peft_model(main_model, peft_config, adapter_name='default')
    main_model.print_trainable_parameters()

    reward_fn = partial(
        compute_reward, 
        reward_model=reward_model, 
        reward_tokenizer=reward_tokenizer, 
        device=device
    )
    reward_fn.__name__ = "compute_reward"
    # разбираемся с логинами
    
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    wandb.login(key= os.environ["WANDB_API_KEY"])
    os.environ["WANDB_DISABLED"] = "false"
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = "RL GPT2"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
    os.environ["WANDB_RUN_NAME"] = os.getenv("WANDB_RUN_NAME") #######################

    create_folders([Path(params.train_gpt.output_dir)]) # Нужно создать папку

    training_args = GRPOConfig(
        learning_rate = params.train_gpt.learning_rate,
        logging_steps = params.train_gpt.logging_steps,
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size = params.train_gpt.per_device_train_batch_size,
        gradient_accumulation_steps = params.train_gpt.gradient_accumulation_steps, 
        num_generations = params.train_gpt.num_generations, 
        max_completion_length = params.train_gpt.max_completion_length,
        max_steps = params.train_gpt.max_steps,
        save_steps = params.train_gpt.save_steps,
        max_grad_norm = params.train_gpt.max_grad_norm,
        report_to = params.train_gpt.report_to, 
        run_name= os.environ["WANDB_RUN_NAME"] 
    )

    trainer = GRPOTrainer(
        model=main_model,
        args = training_args,
        reward_funcs= reward_fn,
        train_dataset=imdb_for_rlhf
    )
    trainer.train()

    wandb.finish()
if __name__ == "__main__":
    train_gpt()
