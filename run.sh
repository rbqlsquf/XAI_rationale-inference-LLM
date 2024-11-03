PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"

$PYTHON_PATH $SOURCE_DIR/inference_pn.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "model/1102+dataup/checkpoint-800" \
    --output_dir "result/1102+dataup/hotpot_tt_800.json"
