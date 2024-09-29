from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json


def create_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


class InferenceInput:
    def __init__(self, _id, input_text, answer):
        self._id = _id
        self.input_text = input_text
        self.answer = answer


def create_example(all_data, instruction, tokenizer):
    all_result = []
    for data in tqdm(all_data):
        data_id = data["_id"]
        Question = data["question"]
        answer = data["answer"]
        context = data["context"]
        supporting_facts = data["supporting_facts"]
        concat_supporting_sent = ""
        document_number_support_sent = ""
        support_dic = {}
        write_supporting_sent = ""
        total_sentence_dic = {}
        # 전체 sentence number 세기 위함
        total_sentence_number = 1
        document_sentence_number = 0
        support_num = []
        supporting_sentence = ""
        document = ""
        for sup_sent in supporting_facts:
            title = sup_sent[0]  # supporting fact의 제목
            set_num = sup_sent[1]
            if title not in support_dic.keys():  # 문장번호
                support_dic[title] = []
            support_dic[title].append(set_num)

        for index, j in enumerate(context):
            title = j[0]
            sentences = ""
            if title in support_dic:
                document_sentence_number = 0
                for sent in j[1]:
                    sentence = "[{}] {}".format(total_sentence_number, sent) + "\n"
                    sentences = sentences + sentence
                    if document_sentence_number in support_dic[title]:
                        support_num.append(total_sentence_number)
                        supporting_sentence = supporting_sentence + sentence
                    document_sentence_number += 1
                    total_sentence_number += 1

            else:
                for sent in j[1]:
                    sentence = "[{}] {}".format(total_sentence_number, sent) + "\n"
                    sentences = sentences + sentence
                    total_sentence_number += 1

            write_sent = "Document {} : {}".format(index + 1, title) + "\n" + "{}".format(sentences)
            document = document + write_sent + "\n"

        prompt = "**Question**\n{}\n\n**Related Document**\n{}\n\n**Output format**\n".format(Question, document)
        response = "**Answer**: {}\n**Supporting Sentences**: {}".format(answer, supporting_sentence)

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": response},
        ]

        all_result.append(
            InferenceInput(
                _id=data_id, input_text=tokenizer.apply_chat_template(messages, tokenize=False), answer=response
            )
        )
        if len(all_result) == 20:
            break
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        # Decode the output tokens back to text

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
            # result["input_text"] = item.input_text
            result["generated_text"] = item.generated_text
            result["answer"] = item.answer
            result["generated_all_answer"] = item.generated_all_answer
            all_result.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    # model_path = "microsoft/phi-2"
    # model_path = "microsoft/Phi-3.5-mini-instruct"
    tokenizer, model = create_model(model_path)

    instruction = (
        instruction
    ) = """**Instruction**
Read the provided **Related Document**, find the **Answer** to the given **Question**, and extract it from the **Related Document**. Additionally, extract the sentence numbers needed to infer the **answer** from the **related document** and write them in the **Supporting Sentences**.  The number of supporting Sentences is at least one. You must strictly follow the **Output format** and provide valid information for all items.

**Output format**
**Answer**: 
**Supporting Sentences**:
"""
    file_path = "data/filtered_dev_data.json"
    batch_size = 16
    print(batch_size)

    with open(file_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    input_data = create_example(dev_data, instruction, tokenizer)

    # Create batches of input items
    batches = list(create_batches(input_data, batch_size))

    answer_batches = generate_batch_answer(batches, tokenizer, model)
    #### 답변작성
    output_path = "qwen_asnwer.json"
    write_result(output_path)
