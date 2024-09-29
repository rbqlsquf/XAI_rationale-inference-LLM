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
from peft import LoraConfig
from trl import SFTTrainer


def create_model(model_path):    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.float16).half()
    tokenizer.padding_side = 'right'
    return tokenizer, model


if __name__ == "__main__":

    model_path = "google/gemma-2b-it"
    tokenizer, model = create_model(model_path)
    data_file = "data/train_data.json"
    eval_data_file = "data/dev_data.json"
    dataset = load_dataset("json", data_files=data_file, split="train")
    eval_dataset = load_dataset("json", data_files=eval_data_file, split="train")
    new_model = "gemma_lora_tuning"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs = 1, #epoch는 1로 설정
        # max_steps=5000, #max_steps을 5000으로 설정
        # 리소스 제약때문에 batch size를 타협해야하는 경우가 발생 -> micro batch size를 줄이고,
        # accumulated step을 늘려, 적절한 size로 gradient를 구해 weight update
        # https://www.youtube.com/watch?v=ptlmj9Y9iwE
        per_device_train_batch_size=4
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        warmup_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=200,
        save_steps=2000,
        do_train = True,
        do_eval = True,
        evaluation_strategy ="steps",
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
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()
    trainer.save_model(new_model)