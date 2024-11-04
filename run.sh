<<<<<<< HEAD
PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
=======
PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"

<<<<<<< HEAD
# $PYTHON_PATH $SOURCE_DIR/inference_mean.py \
#     --model_path $MODEL_DIR/mean/checkpoint-8000 \
#     --output_path $OUTPUT_DIR/mean/hotpot_8000.json


$PYTHON_PATH $SOURCE_DIR/inference_origin.py \
    --model_path $MODEL_DIR/origin/checkpoint-8000 \
    --output_path $OUTPUT_DIR/origin/hotpot_8000.json
=======
# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-1000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_1000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-2000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_2000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-3000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_3000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "model/1103+dataup+loss/checkpoint-3000" \
#     --output_dir "result/1102+dataup/hotpot_tt_3000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "model/1103+dataup+loss/checkpoint-4000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_4000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "model/1103+dataup+loss/checkpoint-5000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_5000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-6000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_6000.json"

# $PYTHON_PATH $SOURCE_DIR/inference_pn.py \
#     --mrc_value "True" \
#     --sum_value "True" \
#     --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-7000" \
#     --output_dir "result/1103+dataup+loss/hotpot_tt_7000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "/hdd/rbqlsquf/1103+dataup+loss/checkpoint-8000" \
    --output_dir "result/1103+dataup+loss/hotpot_tt_8000_test.json"
>>>>>>> 11d7d8a4757072d730dfacc957f0a3763ec1975f
