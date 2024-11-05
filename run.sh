PYTHON_PATH="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM/.venv/bin/python"

BASE_DIR="/home/rbqlsquf2/workspace/XAI_rationale-inference-LLM"
MODEL_DIR="$BASE_DIR/model/loss+dataup+att"
OUTPUT_DIR="$BASE_DIR/result/1103+dataup+loss"
SOURCE_DIR="$BASE_DIR/source"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-9000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_9000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-1000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_1000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-2000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_2000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-3000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_3000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-4000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_4000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-5000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_5000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-6000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_6000.json"

$PYTHON_PATH $SOURCE_DIR/inference_pn_att.py \
    --mrc_value "True" \
    --sum_value "True" \
    --train_model_path "$MODEL_DIR/checkpoint-7000" \
    --output_dir "$OUTPUT_DIR/hotpot_tt_7000.json"