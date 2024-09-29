import os
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from torch.cuda.amp import autocast, GradScaler

def create_model(model_path):    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.float32)
    tokenizer.padding_side = 'left'
    return tokenizer, model


if __name__ == "__main__":

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer, model = create_model(model_path)
    data_file = "data/train_data.json"
    eval_data_file = "data/dev_data.json"
    dataset = load_dataset("json", data_files=data_file, split="train")
    eval_dataset = load_dataset("json", data_files=eval_data_file, split="train")
    new_model = "_lora_tuning"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    peft_params = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    )
    
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs = 1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        warmup_steps=1000,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=200,
        save_steps=2000,
        do_train = True,
        do_eval = True,
        eval_strategy ="steps",
        eval_steps = 2000,
        push_to_hub=False,
        report_to='wandb',
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()
    trainer.save_model(new_model)