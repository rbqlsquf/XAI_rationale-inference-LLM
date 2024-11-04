import os
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
<<<<<<< HEAD
=======
    AutoConfig,
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model
import wandb
from modeling_qwen2_mean import Qwen2ForCausalLM
<<<<<<< HEAD

=======
from torch.nn import functional as F
import argparse

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # sentence_masks를 제외한 features 리스트 생성
        features_without_masks = [{k: v for k, v in f.items() if k != "sent_masks"} for f in features]

        # 부모 클래스에서 features_without_masks 처리
        batch = super().__call__(features_without_masks)

        sentence_masks = [f.get("sent_masks", None) for f in features]
        # sentence_masks가 None이 아닌 경우 패딩 처리
        if sentence_masks[0] is not None:
            max_length = max(len(mask) for mask in sentence_masks)
            padded_sentence_masks = [[0] * (max_length - len(mask)) + mask for mask in sentence_masks]
            batch["sent_masks"] = torch.tensor(padded_sentence_masks)

        return batch
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input을 원하는 대로 수정
        model.model.evidence = None
        # 모델에 수정된 inputs 전달
<<<<<<< HEAD
        outputs = model(**inputs)
        loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss


def create_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(model_path, device_map="cuda")
    new_special_tokens = {"additional_special_tokens": ["<|mrc|>", "<|summary|>"]}
    tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
=======
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if self._is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
            # else:
            #     loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] # path, batch , 1742(max_sent)
        r_loss = loss.requires_grad_(True)
        # r_loss = loss.clone().detach().requires_grad_(True)
        return (r_loss, outputs) if return_outputs else r_loss


def create_model(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, device_map="cuda")
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
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

<<<<<<< HEAD
    example["document"] = example["document"].strip()
    ##############다시
    task_instruction = "Only fill in the **Answer to the **Question based on the **Document if <|MRC|> is True. Do not fill in the **Answer if the Question is not provided or if <|MRC|> is False. Only fill in the **Summary with a summary of the **Document if <|SUM|> is True. Do not fill in the **Summary if <|SUM|> is False."
=======
    task_instruction = "Only fill in the **Answer to the **Question based on the **Document if <|MRC|> is True. Do not fill in the **Answer if the Question is not provided or if <|MRC|> is False. Only fill in the **Summary with a summary of the **Document if <|SUM|> is True. Do not fill in the **Summary if <|SUM|> is False."
    example["document"] = example["document"].strip()
    # token 된 doc
    token_doc = {"input_ids": [], "attention_mask": []}
    # document 문장 index
    sentence_number = 0
    sentence_position = []
    for i, sent in enumerate(example["sent"]):
        # 0번 문장은 instruction으로 지정할 계획
        sent = sent.strip()
        token_sent = tokenizer(sent + " ", add_special_tokens=False)
        sentence_number += 1  # 1부터 시작
        sentence_position.extend([sentence_number] * len(token_sent["input_ids"]))
        token_doc["input_ids"] += token_sent["input_ids"]
        token_doc["attention_mask"] += token_sent["attention_mask"]
    token_end = tokenizer("<|im_end|>\n", add_special_tokens=False)
    sentence_position.extend([0] * len(token_end))
    token_doc["input_ids"] += token_end["input_ids"]
    token_doc["attention_mask"] += token_end["attention_mask"]

>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
    if example["data_type"] == "answer":
        if example["answer_type"] == "F":
            if example["question"] == "no":  # 질문이 없는 경우
                instruction = tokenizer(
<<<<<<< HEAD
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
=======
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
<<<<<<< HEAD
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n{example['document']}<|im_end|>\n",
=======
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
                    add_special_tokens=False,
                )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문은 무조건 있음
            instruction = tokenizer(
<<<<<<< HEAD
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n{example['document']}<|im_end|>\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:{example['output']}\n**Summary:\n<|im_end|>\n",
=======
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:{example['output'].strip()}\n**Summary:\n<|im_end|>\n",
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
                add_special_tokens=False,
            )
    elif example["data_type"] == "summary":
        if example["answer_type"] == "F":  # 무응답의 경우 질문이 무조건 없음
            instruction = tokenizer(
<<<<<<< HEAD
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
=======
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문 유무
            if example["question"] == "summary":  # 질문이 없는 경우
                instruction = tokenizer(
<<<<<<< HEAD
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n{example['document']}<|im_end|>\n",
=======
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
<<<<<<< HEAD
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
=======
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
                    add_special_tokens=False,
                )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:{example['output'].strip()}\n<|im_end|>\n",
                add_special_tokens=False,
            )
    # instruction에 대한 문장 번호
    sentence_position = [0] * len(instruction["input_ids"]) + sentence_position
    sentence_position.extend([0] * len(response["input_ids"]))
    input_ids = instruction["input_ids"] + token_doc["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + token_doc["attention_mask"] + response["attention_mask"]
    labels = [IGNORE_INDEX] * len(instruction["input_ids"] + token_doc["input_ids"]) + response["input_ids"]
    assert len(input_ids) == len(sentence_position) == len(attention_mask) == len(labels)

    if len(input_ids) > MAX_LENGTH:
        sentence_position = sentence_position[:MAX_LENGTH]
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sent_masks": sentence_position,
    }
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f


if __name__ == "__main__":

<<<<<<< HEAD
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer, model = create_model(model_path)
    data_file = "data/train_data_1011.json"

    dataset = Dataset.from_json(data_file)
    dataset = dataset.select(range(12))
    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))

    new_model = "qwen_lora_inst"
=======
    ##############################################################
    #               model param 추가할 내용
    ##############################################################
    parser = argparse.ArgumentParser(description="인자값을 전달받는 Python 스크립트")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_file", type=str, default="data/1022data/hotpot_cnn_6k.json")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_dec_len", type=int, default=1)
    parser.add_argument("--new_model", type=str, default="new_model")
    parser.add_argument("--wandb_project", type=str, default="llm pointer network")
    parser.add_argument("--wandb_run_name", type=str, default="1027")
    parser.add_argument("--output_dir", type=str, default="qwen_lora_1026")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--data_sample", type=bool, default=True)
    args = parser.parse_args()
    print(args)
    #########################################################
    #           변수들 선언
    #########################################################
    model_path = args.model_path

    config = AutoConfig.from_pretrained(model_path)
    config.beam_size = args.beam_size
    config.max_dec_len = args.max_dec_len

    tokenizer, model = create_model(model_path, config)
    data_file = args.data_file
    print("학습 데이터 : ", data_file)
    dataset = Dataset.from_json(data_file)
    if args.data_sample:
        dataset = dataset.select(range(100))
    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))

    new_model = args.new_model
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
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
<<<<<<< HEAD
        if "test" in name:
            param.requires_grad = True
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    wandb.init(project="qwen llm lora")
    wandb.run.name = "1017"
    training_params = TrainingArguments(
        output_dir="qwen_lora_1017",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        logging_steps=10,
        run_name="qwen lora",
=======
        if "gru" in name:
            param.requires_grad = True
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    ##############################################################
    #               wanb
    ##############################################################
    wandb.init(project=args.wandb_project)
    wandb.run.name = args.wandb_run_name

    ##############################################################
    training_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,  # 수정했음
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        logging_steps=1,
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        save_steps=1000,
        save_on_each_node=True,
        do_train=True,
        push_to_hub=False,
        report_to="wandb",
    )
    trainer = CustomTrainer(
        model=model,
        args=training_params,
        train_dataset=processed_dataset,
<<<<<<< HEAD
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
=======
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
    )
    trainer.train()
    trainer.save_model(new_model)
