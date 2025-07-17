# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"



export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="allenai/DataDecide-dclm-baseline-qc-7p-fw2-1B"
bash sh/kaitlyn/eval_datadecide_debug.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH




# # Qwen2.5-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# # Evaluate Qwen2-Math-Instruct
# PROMPT_TYPE="qwen-boxed"

# # Qwen2-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
