from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import torch
from tqdm import tqdm
import json
from peft import PeftModel, PeftConfig
from datasets import Dataset

from modeling_qwen2_pn import Qwen2ForCausalLM
import argparse


def create_model(base_model_path, lora_path,config):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    new_special_tokens = {"additional_special_tokens": ["<|mrc|>", "<|summary|>"]}
    tokenizer.add_special_tokens(new_special_tokens)
    tokenizer.padding_side = "left"
    # config.vocab_size = config.vocab_size + len(new_special_tokens)
    # AutoModelForCausalLM -> Qwen2ForCausalLM
    peft_model = Qwen2ForCausalLM.from_pretrained(lora_path, device_map="auto")
    return tokenizer, peft_model


class InferenceInput:
    def __init__(self, _id, input_text, answer):
        self._id = _id
        self.input_text = input_text
        self.answer = answer


def create_example(all_example, tokenizer):
    all_result = []
    task_instruction = "Only fill in the **Answer to the **Question based on the **Document if <|MRC|> is True. Do not fill in the **Answer if the Question is not provided or if <|MRC|> is False. Only fill in the **Summary with a summary of the **Document if <|SUM|> is True. Do not fill in the **Summary if <|SUM|> is False."
    for example in tqdm(all_example):
        if example["question"] == "summary":
            messages = [
                {"role": "system", "content": f"{task_instruction}\n<|MRC|>True<|SUM|>True"},
                {"role": "user", "content": f"**Document:\n{example['document']}"},
            ]
        else:  # MRC의 경우
            messages = [
                {
                    "role": "system",
                    "content": f"{task_instruction}\n<|MRC|>True<|SUM|>False",
                },
                {"role": "user", "content": f"**Question:{example['question']}\n**Document:\n{example['document']}"},
            ]

        result = {}
        result["input"] = tokenizer.apply_chat_template(messages, tokenize=False)
        result["output"] = example["output"]
        all_result.append(InferenceInput(_id=example["_id"], input_text=result["input"], answer=result["output"]))
        # if len(all_result) == 100:
        #     break
    return all_result


def create_batches(input_list, batch_size):
    # Split the input list into batches of size 'batch_size'
    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]


def generate_batch_answer(batches, tokenizer, model):
    for batch_num, batch in enumerate(tqdm(batches)):
        batch_texts = [item.input_text for item in batch]
        inputs = tokenizer(
            batch_texts,  # Tokenized texts after applying chat template
            return_tensors="pt",  # Return in tensor format
            padding=True,  # Pad sequences to the same length
        ).to("cuda")
        model.to("cuda")
        with torch.no_grad():
            model.model.evidence = None
            outputs = model.generate(
                **inputs,
                sentence_masks=0,  # 여기다가 넣으셈
                max_new_tokens=512,
            )

        decoded_outputs = [
            tokenizer.decode(output[len(inputs[i]) :], skip_special_tokens=True) for i, output in enumerate(outputs)
        ]
        decoded_outputs_ = [tokenizer.decode(output, skip_special_tokens=True) for i, output in enumerate(outputs)]

        # Store the generated text back in the input objects
        for i, item in enumerate(batch):
            item.generated_text = decoded_outputs[i]
            item.generated_all_answer = decoded_outputs_[i]
    return batches


def write_result(output_path):
    all_result = []
    for batch_num, batch in enumerate(answer_batches):
        for item in batch:
            result = {}
            result["_id"] = item._id
            result["input_text"] = item.input_text
            if "assistant" in item.generated_text:
                result["generated_text"] = item.generated_text.split("assistant\n")[1]
            else:
                result["generated_text"] = item.generated_text
            result["answer"] = item.answer
            result["generated_all_answer"] = item.generated_all_answer
            all_result.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="인자값을 전달받는 Python 스크립트")
    # parser.add_argument("--model_path", type=str, required=True, help="모델 경로")
    # parser.add_argument("--output_path", type=str, required=True, help="결과저장 경로")
    # args = parser.parse_args()

    # model_path = args.model_path
    # output_path = args.output_path

    model_path = "/hdd/rbqlsquf/qwen_lora_1026/checkpoint-60"
    output_path = "result/mean/hotpot_1000.json"
    ##########################################

    base_model_path = "Qwen/Qwen2.5-3B-Instruct"
    config = AutoConfig.from_pretrained(base_model_path)
    
    tokenizer, model = create_model(base_model_path, model_path, config)

    file_path = "data/1008data/hotpot_dev.json"
    batch_size = 16
    print(batch_size)

    with open(file_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    input_data = create_example(dev_data, tokenizer)

    # Create batches of input items
    batches = list(create_batches(input_data, batch_size))

    answer_batches = generate_batch_answer(batches, tokenizer, model)
    #### 답변작성

    write_result(output_path)
