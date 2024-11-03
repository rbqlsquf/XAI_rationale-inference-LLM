PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_origin.py \
    --data_file "$BASE_DIR/data/train_hotpot_cnn_1022.json" \
    --new_model "1103+dataup+loss" \
    --output_dir "/hdd/rbqlsquf/1103+dataup+loss" \
    --num_train_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --wandb_run_name "1103+dataup+loss" \
