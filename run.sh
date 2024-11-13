PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="model/1111_1107_yesloss_concat"
OUTPUT_DIR="$BASE_DIR/result/1111_1107_yesloss_concat"
SOURCE_DIR="$BASE_DIR/source"
SCRIPT_NAME="inference_1108.py"

for i in {64..64..2}; do
    checkpoint=$((i * 100))
    $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
        --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
        --output_dir "$OUTPUT_DIR/$checkpoint.json"
done

# for i in {62..68..2}; do
#     checkpoint=$((i * 100))
#     $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
#         --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
#         --output_dir "$OUTPUT_DIR/$checkpoint.json"
# done


# for i in 72; do
#     checkpoint=$((i * 100))
#     $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
#         --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
#         --output_dir "$OUTPUT_DIR/$checkpoint.json"
# done