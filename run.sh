PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"

$PYTHON_PATH $SOURCE_DIR/inference_pn.py \
    --mrc_value "True" \
    --sum_value "True" \
    --output_dir "result/qwen_lora_1031/hotpot_tt.json"
    # --data_file $BASE_DIR/data/1029data/hotpot_dev.json \ 
    # --output_dir $OUTPUT_DIR/qwen_lora_1028/hotpot_1000.json

$PYTHON_PATH $SOURCE_DIR/inference_pn.py \
    --mrc_value "True" \
    --sum_value "False" \
    --output_dir "result/qwen_lora_1031/hotpot_tf.json"

# $PYTHON_PATH $SOURCE_DIR/inference_origin.py \
#     --model_path $MODEL_DIR/origin/checkpoint-8000 \
#     --output_path $OUTPUT_DIR/origin/hotpot_8000.json