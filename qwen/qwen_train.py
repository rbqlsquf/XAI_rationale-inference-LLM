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
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    new_special_tokens = {"additional_special_tokens": ["<|mrc|>", "<|summary|>"]}
    tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"
    return tokenizer, model


if __name__ == "__main__":

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer, model = create_model(model_path)
    data_file = "data/qwen_train_data_only_multi-news.json"
    dataset = load_dataset("json", data_files=data_file, split="train")
    new_model = "lora_tuning_add_new_data_3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    peft_params = LoraConfig(
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_params = TrainingArguments(
        output_dir="./results_500",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=1e-4,
        bf16=False,
        logging_steps=200,
        gradient_checkpointing=True,
        save_steps=1000,
        save_on_each_node=True,
        do_train=True,
        push_to_hub=False,
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()
    trainer.save_model(new_model)
