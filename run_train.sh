PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_pn_noloss.py \
    --data_file "$BASE_DIR/data/train_hotpot_cnn_1022.json" \
    --new_model "1102+dataup" \
    --output_dir "/hdd/rbqlsquf/1102+dataup" \
    --num_train_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --wandb_run_name "1102+dataup" \
