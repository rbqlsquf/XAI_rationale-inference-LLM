import os
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model
import wandb
from modeling_qwen2_pn_att_1107_upper import Qwen2ForCausalLM_pn, BeamSearchAttentionDecoder
from nltk.translate.bleu_score import sentence_bleu
from torch.nn import functional as F
import argparse
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # sentence_masks를 제외한 features 리스트 생성
        features_without_masks = [
            {k: v for k, v in f.items() if k != "sent_masks" and k != "gold_sp"} for f in features
        ]
        # 부모 클래스에서 features_without_masks 처리
        batch = super().__call__(features_without_masks)

        sentence_masks = [f.get("sent_masks", None) for f in features]
        gold_sp = [f.get("gold_sp", None) for f in features]
        # sentence_masks가 None이 아닌 경우 패딩 처리
        if sentence_masks[0] is not None:
            max_length = max(len(mask) for mask in sentence_masks)
            padded_sentence_masks = [[0] * (max_length - len(mask)) + mask for mask in sentence_masks]
            batch["sent_masks"] = torch.tensor(padded_sentence_masks)
        if gold_sp[0] is not None:
            max_length = 3
            padded_sentence_masks = []
            for sp in gold_sp:
                if len(sp) > max_length:
                    sp = sp[:max_length]
                # Pad if shorter than max_length
                padded_sp = sp + [0] * (max_length - len(sp))
                padded_sentence_masks.append(padded_sp)
            batch["gold_sp"] = torch.tensor(padded_sentence_masks)
        return batch


class CustomTrainer(Trainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        self.model.model.save_pn_model(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        # input을 원하는 대로 수정
        model.model.evidence = None

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
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]  # path, batch , 1742(max_sent)

        sampled_evidence_scores = outputs.get("attention_scores")  # batch*path, 2, max_sent??
        mask = outputs.get("mask")  # batch, dec_len, max_sent
        path_logits = outputs.get("path_logits")  # path, batch, max_len, 151667
        sampled_evidence_sentence = outputs.get("evidence_sentences")

        ###############
        loss_fct_2 = CrossEntropyLoss()
        loss_2 = loss_fct_2(
            sampled_evidence_scores.view(-1, sampled_evidence_scores.size(-1)), inputs["gold_sp"].view(-1)
        )

        r_loss = (loss[0, :].mean() + loss_2) / 2
        print("========================================")
        print(self.state.global_step)
        print("loss:{}".format(loss))
        print("loss_mean:{}".format(loss[0, :].mean()))
        print("loss_2:{}".format(loss_2))
        print("r_loss : {}".format(r_loss))
        # Add wandb logging for the evidence losses
        # Detailed wandb logging
        return (r_loss, outputs) if return_outputs else r_loss


def create_model(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM_pn.from_pretrained(model_path, config=config, device_map="cuda")
    model.enable_input_require_grads()
    gru = BeamSearchAttentionDecoder(
        hidden_size=config.hidden_size, num_sent=config.max_dec_len, topk=config.beam_size
    )
    model.set_gru(gru)
    model.config.use_cache = False
    tokenizer.padding_side = "left"
    return tokenizer, model


def create_model_for_debug(base_model_path, lora_path, config):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    trained_model = Qwen2ForCausalLM_pn.from_pretrained(lora_path, config=config, device_map="auto")
    gru = BeamSearchAttentionDecoder(
        hidden_size=config.hidden_size, num_sent=config.max_dec_len, topk=config.beam_size
    )
    trained_model.set_gru(gru)
    trained_model.config.use_cache = False
    tokenizer.padding_side = "left"
    print("LORA WEIGHT LOADING")
    trained_model.load_pn_model(lora_path)
    return tokenizer, trained_model


IGNORE_INDEX = -100


def process_func(example, tokenizer):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
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

    ########################################################################################################################
    #           전처리 형태 바꾸기
    ########################################################################################################################
    instruction = tokenizer(
        f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n**Question:{example['question']}\n**Document:\n",
        add_special_tokens=False,
    )
    response = tokenizer(
        f"<|im_start|>assistant\n**Answer:{example['output'].strip()}<|im_end|>\n", add_special_tokens=False
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
        "gold_sp": example["supporting_num"],
    }


if __name__ == "__main__":

    ##############################################################
    #               model param 추가할 내용
    ##############################################################
    parser = argparse.ArgumentParser(description="인자값을 전달받는 Python 스크립트")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_file", type=str, default="data/1113data/hotpot_train.json")
    parser.add_argument("--lora_path", type=str, default="model/1112_yesloss/checkpoint-1000")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_dec_len", type=int, default=3)
    parser.add_argument("--new_model", type=str, default="new_mode")
    parser.add_argument("--wandb_project", type=str, default="llm pointer network")
    parser.add_argument("--wandb_run_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="qwen_lora_1026")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--data_sample", type=bool, default=False)
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
    # tokenizer, model = create_model_for_debug(model_path, args.lora_path, config)
    data_file = args.data_file
    print("학습 데이터 : ", data_file)
    dataset = Dataset.from_json(data_file)
    if args.data_sample:
        dataset = dataset.select(range(10))
    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))

    new_model = args.new_model
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
        if "gru" in name or "linear_w1" in name:
            param.requires_grad = True
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    ##############################################################
    #               wanb
    ##############################################################
    wandb.init(project=args.wandb_project, save_code=True)
    wandb.run.name = args.wandb_run_name
    wandb.save("modeling_qwen2_pn_att_1107.py")
    ##############################################################
    training_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,  # 수정했음
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        logging_steps=1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        save_steps=200,
        save_on_each_node=True,
        do_train=True,
        push_to_hub=False,
        report_to="wandb",
    )
    trainer = CustomTrainer(
        model=model,
        args=training_params,
        train_dataset=processed_dataset,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(new_model)
