PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="model/1112_yesloss"
OUTPUT_DIR="$BASE_DIR/result/1112_yesloss"
SOURCE_DIR="$BASE_DIR/source"
SCRIPT_NAME="inference_1108.py"

for i in {2..12..2}; do
    checkpoint=$((i * 100))
    $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
        --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
        --output_dir "$OUTPUT_DIR/$checkpoint.json"
done


# for i in 1 3 7; do
#     checkpoint=$((i * 100))
#     $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
#         --data_file "data/1029data/cnn_dev.json" \
#         --mrc_value "False" \
#         --sum_value "True" \
#         --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
#         --output_dir "$OUTPUT_DIR/hotpot_ft_$checkpoint.json"
# done