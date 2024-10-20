PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"

$PYTHON_PATH $SOURCE_DIR/inference_mean.py \
    --model_path $MODEL_DIR/mean/checkpoint-8000 \
    --output_path $OUTPUT_DIR/mean/hotpot_8000.json


# $PYTHON_PATH $SOURCE_DIR/inference_origin.py \
#     --model_path $MODEL_DIR/origin/checkpoint-8000 \
#     --output_path $OUTPUT_DIR/origin/hotpot_8000.json