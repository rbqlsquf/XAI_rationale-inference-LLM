from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import torch
from tqdm import tqdm
import json
from peft import PeftModel, PeftConfig
from datasets import Dataset

from modeling_qwen2_pn_att import Qwen2ForCausalLM_pn, BeamSearchAttentionDecoder
import argparse


def create_model(base_model_path, lora_path, config):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    trained_model = Qwen2ForCausalLM_pn.from_pretrained(lora_path, config=config, device_map="auto")
    gru = BeamSearchAttentionDecoder(hidden_size=config.hidden_size, num_sent=config.max_dec_len, topk=config.beam_size)
    trained_model.set_gru(gru)
    trained_model.config.use_cache = False
    tokenizer.padding_side = "left"
    trained_model.load_pn_model(lora_path)
    return tokenizer, trained_model


class InferenceInput:
    def __init__(self, _id, input_text, answer, attention_mask, sent_masks, gold_sp):
        self._id = _id
        self.input_text = input_text
        self.answer = answer
        self.attention_mask = attention_mask
        self.sent_masks = sent_masks
        self.gold_sp = gold_sp


def create_example(all_example, tokenizer, data_sample, mrc_value, sum_value):
    all_result = []

    task_instruction = "Only fill in the **Answer to the **Question based on the **Document if <|MRC|> is True. Do not fill in the **Answer if the Question is not provided or if <|MRC|> is False. Only fill in the **Summary with a summary of the **Document if <|SUM|> is True. Do not fill in the **Summary if <|SUM|> is False."
    for example in tqdm(all_example):
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

        if example["question"] == "summary":
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Document:\n",
                add_special_tokens=False,
            )
            # response = tokenizer(
            #     f"<|im_start|>assistant\n**Answer:\n**Summary:{example['output'].strip()}\n<|im_end|>\n",
            #     add_special_tokens=False,
            # )
            response = f"<|im_start|>assistant\n**Answer:\n**Summary:{example['output'].strip()}\n<|im_end|>\n"
        else:  # MRC의 경우
            instruction = tokenizer(
                f"<|im_start|>system\n{task_instruction}\n<|MRC|>{mrc_value}<|SUM|>{sum_value}<|im_end|>\n<|im_start|>user\n**Question:{example['question'].strip()}\n**Document:\n",
                add_special_tokens=False,
            )
            # response = tokenizer(
            #     f"<|im_start|>assistant\n**Answer:{example['output'].strip()}\n**Summary:\n<|im_end|>\n",
            #     add_special_tokens=False,
            # )
            response = f"<|im_start|>assistant\n**Answer:{example['output'].strip()}\n**Summary:\n<|im_end|>\n"
        sentence_position = [0] * len(instruction["input_ids"]) + sentence_position
        input = instruction["input_ids"] + token_doc["input_ids"]
        attention_mask = instruction["attention_mask"] + token_doc["attention_mask"]
        output = response

        if "supporting_num" in example.keys():
            gold_sp = example["supporting_num"]
        else:
            gold_sp = None
        assert len(input) == len(sentence_position) == len(attention_mask)

        all_result.append(
            InferenceInput(
                _id=example["_id"],
                input_text=input,
                answer=output,
                attention_mask=attention_mask,
                sent_masks=sentence_position,
                gold_sp=gold_sp,
            )
        )
        if data_sample:
            if len(all_result) == 100:
                break
    return all_result


def create_batches(input_list, batch_size):
    # Split the input list into batches of size 'batch_size'
    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]


def generate_batch_answer(batches, tokenizer, model):
    for batch_num, batch in enumerate(tqdm(batches)):
        input_ids = [item.input_text for item in batch]
        attention_mask = [item.attention_mask for item in batch]
        sentence_masks = [item.sent_masks for item in batch]

        model.to("cuda")
        input_batch = {}
        max_length = max(len(mask) for mask in input_ids)
        padded_input_ids = [[tokenizer.pad_token_id] * (max_length - len(mask)) + mask for mask in input_ids]
        input_batch["input_ids"] = torch.tensor(padded_input_ids).cuda()
        padded_attention_mask = [[0] * (max_length - len(mask)) + mask for mask in attention_mask]
        input_batch["attention_mask"] = torch.tensor(padded_attention_mask).cuda()
        padded_sentence_masks = [[0] * (max_length - len(mask)) + mask for mask in sentence_masks]
        input_batch["sent_masks"] = torch.tensor(padded_sentence_masks).cuda()

        with torch.no_grad():
            model.evidence = None
            model.sentence_number = None
            outputs = model.generate(
                input_ids=input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
                sent_masks=input_batch["sent_masks"],
                max_new_tokens=200,
            )

        input_text = [tokenizer.decode(input_id, skip_special_tokens=True) for i, input_id in enumerate(input_ids)]
        decoded_outputs = [
            tokenizer.decode(output[len(input_text) :], skip_special_tokens=True) for i, output in enumerate(outputs)
        ]
        decoded_outputs_ = [tokenizer.decode(output, skip_special_tokens=True) for i, output in enumerate(outputs)]

        # Store the generated text back in the input objects
        for i, item in enumerate(batch):
            item.input_text = input_text
            item.generated_text = decoded_outputs[i]
            item.generated_all_answer = decoded_outputs_[i]
            if model.sentence_number != None:
                item.pred_sp = model.sentence_number[i]
    return batches


def write_result(output_path, answer_batches, tokenizer):
    all_result = []
    for batch_num, batch in enumerate(answer_batches):
        for item in batch:
            result = {}
            result["_id"] = item._id
            if "assistant\n" in item.generated_text:
                result["generated_text"] = item.generated_text.split("assistant\n")[1]
            elif "assistant" in item.generated_text:
                result["generated_text"] = item.generated_text.split("assistant")[1]
            else:
                result["generated_text"] = item.generated_text
            result["answer"] = item.answer
            result["generated_all_answer"] = item.generated_all_answer
            if item.gold_sp != None:
                result["gold_sp"] = item.gold_sp
                result["pred_sp"] = item.pred_sp.tolist()
            all_result.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    ##############################################################
    #               model param 추가할 내용
    ##############################################################
    parser = argparse.ArgumentParser(description="인자값을 전달받는 Python 스크립트")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_model_path", type=str, default="model/1105/checkpoint-2000")
    parser.add_argument("--data_file", type=str, default="data/1029data/hotpot_dev_supporting.json")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_dec_len", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="result/1105/hotpot_tt_2000.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_sample", type=bool, default=True)
    parser.add_argument("--mrc_value", type=str, default=True)
    parser.add_argument("--sum_value", type=str, default=True)
    args = parser.parse_args()
    print(args)
    #########################################################
    #           변수들 선언
    #########################################################

    config = AutoConfig.from_pretrained(args.base_model_path)
    config.beam_size = args.beam_size
    config.max_dec_len = args.max_dec_len

    tokenizer, model = create_model(args.base_model_path, args.train_model_path, config)
    print("batch size : ", args.batch_size)

    with open(args.data_file, "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    input_data = create_example(dev_data, tokenizer, args.data_sample, args.mrc_value, args.sum_value)

    # Create batches of input items
    batches = list(create_batches(input_data, args.batch_size))

    answer_batches = generate_batch_answer(batches, tokenizer, model)
    #### 답변작성

    write_result(args.output_dir, answer_batches, tokenizer)