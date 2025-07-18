set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval_debug

SPLIT="test"
NUM_TEST_SAMPLE=5  # just 5 samples for debug

# simple debug - just gsm8k
DATA_NAME="gsm8k"
TOKENIZERS_PARALLELISM=false \
# python3.8 -u math_eval.py \
uv run python math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite 