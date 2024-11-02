PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


# $PYTHON_PATH $SOURCE_DIR/train_pn.py \
#     --data_file "$BASE_DIR/data/1022data/hotpot_cnn_6k.json" \
#     --new_model "qwen_lora_1101" \
#     --output_dir "/hdd/rbqlsquf/qwen_lora_1102+loss" \
#     --num_train_epochs 1 \
#     --batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --wandb_run_name "1102" \

$PYTHON_PATH $SOURCE_DIR/train_origin.py \