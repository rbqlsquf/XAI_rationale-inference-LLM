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

from peft import LoraConfig, get_peft_model
import wandb
from modeling_qwen2_pn import Qwen2ForCausalLM_pn


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


class CustomTrainer(Trainer):
    def generate_sentences(
        self, model, inputs, r_batch_size, loss, sampled_evidence_scores, mask, logits, sampled_evidence_sentence
    ):
        # [batch, beam_size, max_dec, -1]
        predicted_answer = []
        evidence_predicted_answer = []
        for path in range(config.beam_size):
            logit = torch.argmax(logits[path], dim=-1)

            decoded_outputs = [
                tokenizer.decode(output[inputs["labels"][i] != -100], skip_special_tokens=True)
                for i, output in enumerate(logit)
            ]
            predicted_answer.append(decoded_outputs)
            #####################################################
            #           근거 문장 생성
            #####################################################
            evidence_path = (
                sampled_evidence_sentence.view(r_batch_size, -1, config.max_dec_len).transpose(0, 1)[path].tolist()
            )
            #####################################################
            #           근거 문장에 따른 입력 재구성
            #####################################################
            sample_inputs = []
            for k in range(r_batch_size):
                # 첫 번째 1의 인덱스 찾기

                #############################################################
                #                   sentence_group에 대한 내용
                #############################################################
                sentence_groups = {}
                for idx, sentence_num in enumerate(inputs["sent_masks"][k]):
                    if str(sentence_num) not in sentence_groups:
                        sentence_groups[str(sentence_num)] = []
                    sentence_groups[str(sentence_num)].append(idx)

                first_one_index = (inputs["sent_masks"][k] == 1).nonzero(as_tuple=True)[0][0].item()
                tmp_sentence_mask = [0] * len(inputs["sent_masks"][k][:first_one_index])
                see_tokens = list(range(0, len(inputs["sent_masks"][k][:first_one_index])))
                for j in range(config.max_dec_len):
                    see_tokens.extend(
                        (inputs["sent_masks"][k] == evidence_path[k][j]).nonzero(as_tuple=True)[0].tolist()
                    )

                    tmp_sentence_mask = tmp_sentence_mask + [j + 1] * len(
                        (inputs["sent_masks"][k] == evidence_path[k][j]).nonzero(as_tuple=True)[0].tolist()
                    )
                sentences = inputs["input_ids"][k][see_tokens]
                tmp_input_ids = sentences.tolist()
                tmp_input_ids = tmp_input_ids + [tokenizer.eos_token_id] + tokenizer.encode("\n")
                tmp_sentence_mask.extend([0] * 2)
                tokens = tokenizer.decode(tmp_input_ids)
                tmp_attention_mask = torch.ones(len(tmp_input_ids), dtype=torch.long).tolist()

                assert len(tmp_input_ids) == len(tmp_attention_mask) == len(tmp_sentence_mask)
                # 데이터 추가하는 방법

                tmp_input_ids = tmp_input_ids

                sample_input = {
                    "input_ids": tmp_input_ids,
                }
                sample_inputs.append(sample_input)
            ###############################################
            input_ids = [f.get("input_ids", None) for f in sample_inputs]
            max_length = max(len(mask) for mask in input_ids)
            batch = {}

            padded_input_ids = [[tokenizer.pad_token_id] * (max_length - len(mask)) + mask for mask in input_ids]
            batch["input_ids"] = torch.tensor(padded_input_ids).cuda()
            ####################################################################
            e_outputs = model.generate(batch["input_ids"], max_new_tokens=5)  #!!!바꾸기
            e_decoded_outputs = [
                tokenizer.decode(output[len(batch["input_ids"][i]) :], skip_special_tokens=True)
                for i, output in enumerate(e_outputs)
            ]

            #####################################################
            #           근거 문장만 입력으로 했을 때 출력 친구들 넣어주기
            #####################################################
            evidence_predicted_answer.append(e_decoded_outputs)
        #########################################
        #       beam size에 대해서 predict_answer, evidence_predicted_answer 완료
        #########################################
        return predicted_answer, evidence_predicted_answer

    def compute_evidence_f1_score(self, predicted_answer, evidence_predicted_answer, inputs):
        # [path, batch]
        f1_list = [1e-3 for _ in range(config.beam_size)]
        g_f1_list = [1e-3 for _ in range(config.beam_size)]

        filtered_labels = [labels[labels != -100].tolist() for labels in inputs["labels"]]
        gold_list = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
        for path in range(config.beam_size):
            predicted = predicted_answer[path]
            e_predicted = evidence_predicted_answer[path]

        return 0

    def compute_loss(self, model, inputs, return_outputs=False):
        # input을 원하는 대로 수정
        model.model.evidence = None
        # 모델에 수정된 inputs 전달
        outputs = model(**inputs)
        loss = outputs.get("loss")  # path, batch , 1742(max_sent)
        sampled_evidence_scores = outputs.get("attention_scores")  # batch*path, 2, max_sent??
        mask = outputs.get("mask")  # batch, dec_len, max_sent
        path_logits = outputs.get("path_logits")  # path, batch, max_len, 151667
        sampled_evidence_sentence = outputs.get("evidence_sentences")
        #####################################################################
        #               형태 바꾸기
        #####################################################################
        r_batch_size = mask.size(0)
        sampled_evidence_scores = sampled_evidence_scores.view(r_batch_size, config.beam_size, config.max_dec_len, -1)
        #####################################################################
        #              먼저 답변 부터 생성
        #####################################################################
        predicted_answer, evidence_predicted_answer = self.generate_sentences(
            model, inputs, r_batch_size, loss, sampled_evidence_scores, mask, path_logits, sampled_evidence_sentence
        )

        best_path = self.compute_evidence_f1_score(predicted_answer, evidence_predicted_answer, inputs)

        return (loss, outputs) if return_outputs else loss


def create_model(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM_pn.from_pretrained(model_path, config=config, device_map="cuda")
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
    sentence_position.extend([sentence_number] * len(token_end))
    token_doc["input_ids"] += token_end["input_ids"]
    token_doc["attention_mask"] += token_end["attention_mask"]

    if example["data_type"] == "answer":
        if example["answer_type"] == "F":
            if example["question"] == "no":  # 질문이 없는 경우
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
                    add_special_tokens=False,
                )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문은 무조건 있음
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:{example['output'].strip()}\n**Summary:\n<|im_end|>\n",
                add_special_tokens=False,
            )
    elif example["data_type"] == "summary":
        if example["answer_type"] == "F":  # 무응답의 경우 질문이 무조건 없음
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
                add_special_tokens=False,
            )
            response = tokenizer(
                f"<|im_start|>assistant\n**Answer:\n**Summary:\n<|im_end|>\n", add_special_tokens=False
            )
        else:  # 답 해야하는 경우 질문 유무
            if example["question"] == "summary":  # 질문이 없는 경우
                instruction = tokenizer(
                    f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
                    add_special_tokens=False,
                )
            else:
                instruction = tokenizer(
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


if __name__ == "__main__":

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    config = AutoConfig.from_pretrained(model_path)
    ##############################################################
    #               model param 추가할 내용
    ##############################################################
    config.beam_size = 3
    config.max_dec_len = 5

    tokenizer, model = create_model(model_path, config)
    data_file = "data/1020data/train_data_1022.json"

    dataset = Dataset.from_json(data_file)
    # 아래 코드는 일부만 가지고 오기 위함
    dataset = dataset.select(range(100))
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

    ##############################################################
    #               wanb
    ##############################################################
    wandb.init(project="qwen llm lora")
    wandb.run.name = "1017"

    ##############################################################
    training_params = TrainingArguments(
        output_dir="qwen_lora_2020",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # 수정했음
        gradient_accumulation_steps=1,
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
    trainer = CustomTrainer(
        model=model,
        args=training_params,
        train_dataset=processed_dataset,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(new_model)
