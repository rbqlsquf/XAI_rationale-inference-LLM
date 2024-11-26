import re


def normalize_answer(s):
    """간단한 토큰화와 정규화"""
    s = s.lower()  # 소문자 변환
    s = re.sub(r"\b(a|an|the)\b", " ", s)  # 불필요한 관사 제거
    s = re.sub(r"[^a-z0-9]", " ", s)  # 알파벳과 숫자 외 제거
    return " ".join(s.split())  # 공백 정리


def exact_match_score(prediction, ground_truth):
    """예측 답과 실제 답 간의 EM 점수 계산"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_hotpot(prediction, ground_truth):
    """예측 답과 실제 답 간의 F1 점수 계산"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        return 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_supporting_facts(gold_sp, pred_sp):
    """Supporting facts에 대한 EM, Precision, Recall, F1 점수를 계산하는 함수"""
    # 단일 정수를 리스트로 변환
    gold_sp = [gold_sp] if isinstance(gold_sp, int) else gold_sp
    pred_sp = [pred_sp] if isinstance(pred_sp, int) else pred_sp

    # 예측과 정답 집합으로 변환
    gold_set = set(gold_sp)
    pred_set = set(pred_sp)

    # True Positives 계산
    tp = len(gold_set & pred_set)

    # Precision, Recall 계산
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0

    # F1 점수 계산
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Exact Match 계산
    em = 1 if gold_set == pred_set else 0

    return em, precision, recall, f1


import json

file_path = "data/1125data/hotpot_dev.json"
with open(file_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

for i in range(30, 31, 2):
    f_name = f"result/1126_upper/{i}00.json"

    with open(f_name, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    score = []
    all_em_score = []
    all_precision_score = []
    all_recall_score = []
    all_f1_score = []
    result_f1 = []
    result_em = []
    ignore = 0
    for dev, data in zip(dev_data, test_data):
        assert dev["_id"] == data["_id"]
        predict = ""
        answer = (
            data["answer"]
            .replace("**Answer:", "")
            .replace("<|im_start|>assistant", "")
            .replace("<|im_end|>", "")
            .strip()
        )
        generated_text = data["generated_text"].replace("**Answer:", "").strip()
        if answer == "yes":
            if answer in generated_text.lower() and "no" not in generated_text.lower():
                generated_text = "yes"
            else:
                generated_text = ""
        elif answer == "no":
            if answer in generated_text.lower() and "yes" not in generated_text.lower():
                generated_text = "no"
            else:
                generated_text = ""
        answer = answer.strip()
        predict = generated_text.strip()
        print(answer)
        print(generated_text)
        print("==========================")
        result_f1.append(f1_score_hotpot(answer, predict))
        result_em.append(exact_match_score(predict, answer))
        ################################################
        gold_sp = data["gold_sp"]
        # pred_sp = data["pred_sp"]
        pred_sp = [x for x in data["pred_sp"] if x != 0]
        em, precision, recall, f1 = evaluate_supporting_facts(gold_sp, pred_sp)
        all_em_score.append(em)
        all_precision_score.append(precision)
        all_recall_score.append(recall)
        all_f1_score.append(f1)

        for i in pred_sp:
            if answer == "yes" or answer == "no":
                ignore += 1
                break
            if predict in dev["sent"][i - 1]:
                score.append(dev["_id"])
                # print(answer)
                # print(generated_text)
                # print(dev["sent"][i-1])
                # print("================")
                break

        # F1 점수와 EM 점수 출력
    print(f_name)
    print("F1 점수: ", sum(result_f1) / len(result_f1))
    print("EM 점수: ", sum(result_em) / len(result_em))

    # F1 점수와 EM 점수 출력
    print("all_em_score 점수: ", sum(all_em_score) / len(all_em_score))
    print("all_f1_score 점수: ", sum(all_f1_score) / len(all_f1_score))
    print("all_precision_score 점수: ", sum(all_precision_score) / len(all_precision_score))
    print("all_recall_score 점수: ", sum(all_recall_score) / len(all_recall_score))
    print("=================================================")
    print(len(result_em))
