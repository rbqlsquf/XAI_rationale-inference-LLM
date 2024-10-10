import os
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from torch.cuda.amp import autocast, GradScaler
import wandb


def create_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    new_special_tokens = {"additional_special_tokens": ["<|mrc|>", "<|summary|>"]}
    tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()
    model.config.use_cache = False
    tokenizer.padding_side = "left"
    return tokenizer, model


IGNORE_INDEX = -100


def process_func(example, tokenizer):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    if example["question"] != "summary":
        instruction = tokenizer(
            f"<|im_start|>system\n<|MRC|>True<|SUM|>False<|im_end|>\n<|im_start|>user\n{example['question']}\n{example['document']}<|im_end|>\n",
            add_special_tokens=False,
        )  # add_special_tokens 不在开头加 special_tokens
    else:
        instruction = tokenizer(
            f"<|im_start|>system\n<|MRC|>False<|SUM|>True<|im_end|>\n<|im_start|>user\n{example['document']}<|im_end|>\n",
            add_special_tokens=False,
        )  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"<|im_start|>assistant\n{example['output']}<|im_end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [IGNORE_INDEX] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer, model = create_model(model_path)
    data_file = "data/train_data_1008.json"

    dataset = Dataset.from_json(data_file)

    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))

    new_model = "lora_tuning_re"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    peft_config = LoraConfig(
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    wandb.init(project="qwen lora")
    training_params = TrainingArguments(
        output_dir="./1008",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        logging_steps=10,
        run_name="qwen lora",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        save_steps=1000,
        save_on_each_node=True,
        do_train=True,
        push_to_hub=False,
        report_to="wandb",
    )
    trainer = Trainer(
        model=model,
        args=training_params,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(new_model)
