PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_pn.py \
    --data_file $BASE_DIR/data/1020data/train_hotpot_cnn_filtered.json \
    --new_model /hdd/rbqlsquf/loss+dataup+att \
    --output_dir /hdd/rbqlsquf/loss+dataup+att \
    --num_train_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --wandb_run_name loss+dataup+att
