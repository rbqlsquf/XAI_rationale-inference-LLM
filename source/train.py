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
from modeling_qwen2_pn import Qwen2ForCausalLM


def create_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(model_path, device_map="cuda")
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
    mrc_value = -1
    sum_value = -1
    if example["mrc_type"] == "T":
        mrc_value = "True"
    else:
        mrc_value = "False"
    if example["sum_type"] == "T":
        sum_value = "True"
    else:
        sum_value = "False"

    example["document"] = example["document"].strip()
    ##############다시
    task_instruction = "Only fill in the **Answer to the **Question based on the **Document if <|MRC|> is True. Do not fill in the **Answer if the Question is not provided or if <|MRC|> is False. Only fill in the **Summary with a summary of the **Document if <|SUM|> is True. Do not fill in the **Summary if <|SUM|> is False."
    if example["data_type"] == "answer":
        if example["answer_type"] == "F":
            if example["question"] == "no":  # 질문이 없는 경우
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n{example['document']}<|im_end|>\n",
                    add_special_tokens=False,
                )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문은 무조건 있음
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n{example['document']}<|im_end|>\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:{example['output']}\n**Summary:\n<|im_end|>\n",
                add_special_tokens=False,
            )
    elif example["data_type"] == "summary":
        if example["answer_type"] == "F":  # 무응답의 경우 질문이 무조건 없음
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문 유무
            if example["question"] == "summary":  # 질문이 없는 경우
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n{example['document']}<|im_end|>\n",
                    add_special_tokens=False,
                )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:{example['output']}\n<|im_end|>\n",
                add_special_tokens=False,
            )

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
    data_file = "data/1010data/train_data_1011.json"

    dataset = Dataset.from_json(data_file)

    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))

    new_model = "qwen_lora_inst"
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
        if "test" in name:
            param.requires_grad = True
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    wandb.init(project="qwen llm lora")
    wandb.run.name = "1016"
    training_params = TrainingArguments(
        output_dir="1015",
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
