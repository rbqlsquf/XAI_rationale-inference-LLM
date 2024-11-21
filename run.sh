PYTHON_PATH="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="/hdd/rbqlsquf/1119_2step"
OUTPUT_DIR="$BASE_DIR/result/1119_2step"
SOURCE_DIR="$BASE_DIR/source"
SCRIPT_NAME="inference_1108.py"

for i in {82..2..-2}; do
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


# for i in 54 68 70; do
#     checkpoint=$((i * 100))
#     $PYTHON_PATH $SOURCE_DIR/$SCRIPT_NAME \
#         --train_model_path "$MODEL_DIR/checkpoint-$checkpoint" \
#         --output_dir "$OUTPUT_DIR/$checkpoint.json"
# done