PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_pn_noloss_1108.py \
    --new_model 1110_1106_concat \
    --output_dir model/1110_1106_concat \
    --num_train_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --wandb_run_name 1110_1106_concat
