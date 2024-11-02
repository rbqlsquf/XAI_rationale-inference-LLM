PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model"
OUTPUT_DIR="$BASE_DIR/result"
SOURCE_DIR="$BASE_DIR/source"


$PYTHON_PATH $SOURCE_DIR/train_pn.py \
    --data_file "$BASE_DIR/data/1022data/hotpot_cnn_6k.json" \
<<<<<<< HEAD
    --new_model "qwen_lora_1101" \
    --output_dir "/hdd/rbqlsquf/qwen_lora_1101" \
=======
    --new_model "qwen_lora_1101+loss_" \
    --output_dir "qwen_lora_1101+loss" \
>>>>>>> e4d0148596066e1c1dd7e2884f45b76246d94394
    --num_train_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
<<<<<<< HEAD
    --wandb_run_name "1101" \
=======
    --wandb_run_name "1031+loss" \
>>>>>>> e4d0148596066e1c1dd7e2884f45b76246d94394
