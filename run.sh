PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model/1106_weighted_rationale+noloss"
OUTPUT_DIR="$BASE_DIR/result/1106_weighted_rationale+noloss"
SOURCE_DIR="$BASE_DIR/source"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-2000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_2000.json"
