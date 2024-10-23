import re
import json
from sklearn.metrics import f1_score


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


if __name__ == "__main__":

    with open("result/hotpot_cnn/hotpot_3000_tt.json", "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    result_f1 = []
    result_em = []
    for dev in dev_data:
        predict = ""
        answer = dev["answer"]
        generated_text = dev["generated_text"].split("**Summary:")[0].replace("**Answer:", "")
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
        print(predict)
        print("----")
        result_f1.append(f1_score_hotpot(answer, predict))
        result_em.append(exact_match_score(predict, answer))

        # F1 점수와 EM 점수 출력
    print("F1 점수: ", sum(result_f1) / len(result_f1))
    print("EM 점수: ", sum(result_em) / len(result_em))
