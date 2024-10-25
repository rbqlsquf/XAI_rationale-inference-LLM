PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_pn.py \
    --data_file $BASE_DIR/data/hotpot_cnn_6k.json \
    --new_model qwen_lora_1025 \
    --output_dir /hdd/rbqlsquf/qwen_lora_1025 \
    --num_train_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1