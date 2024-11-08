import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

    
def calculate_bleu(reference, candidate):
    # 문장을 토큰화
    reference_tokens = [word_tokenize(reference.lower())]
    candidate_tokens = word_tokenize(candidate.lower())
    
    # BLEU 점수 계산 (1-gram부터 4-gram까지의 누적 점수)
    weights = (1, 0, 0, 0)  # unigram에만 가중치 부여
    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights)

for i in [1,3,7]:
    file_path = f"result/1106_weighted_sum/hotpot_ft_{i}000.json"
    with open(file_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)
        
    bleu_scores = []
    for dev in dev_data:
        predict = ""
        answer = dev["answer"].split("**Summary:")[1].replace("\n<|im_end|>", "").strip()
        if "**Summary:" in dev["generated_text"]:
            predict = dev["generated_text"].split("**Summary:")[1].strip()
        else:
            predict = dev["generated_text"]
        
        print(answer)
        print("--")
        print(predict)
        print("=============")
        
        bleu_score = calculate_bleu(answer, predict)
        bleu_scores.append(bleu_score)

    # 평균 BLEU 점수 계산
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(file_path)
    print(f"Average BLEU score: {average_bleu:.4f}")
