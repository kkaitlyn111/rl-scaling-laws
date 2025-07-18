# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"



# export CUDA_VISIBLE_DEVICES="0"

# export XDG_CACHE_HOME=/juice5b/scr5b/kaitwang/.cache
# export XDG_CONFIG_HOME=/juice5b/scr5b/kaitwang/.config
# export XDG_DATA_HOME=/juice5b/scr5b/kaitwang/.local/share
# export XDG_STATE_HOME=/juice5b/scr5b/kaitwang/.local/state
# export UV_CACHE_DIR=/juice5b/scr5b/kaitwang/.uv_cache

# export PROJHOME=/juice5b/scr5b/kaitwang/projects
# export PROJDATA=/juice5b/scr5b/kaitwang/data
# export PATH=$PATH:/usr/local/cuda:hipconfig:/u/nlp/bin
# export PYTHONPATH='.'
# export HUGGINGFACE_HUB_CACHE=/juice5b/scr5b/kaitwang/cache/huggingface/hub
# export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/cache/huggingface/datasets
# export WANDB_CACHE_DIR=/juice5b/scr5b/kaitwang/cache/wandb
# export CUDA_ROOT=/usr/local/cuda
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/cuda/targets/x86_64-linux/include
# export WANDB_DIR=/juice5b/scr5b/kaitwang/wandb
# export TMPDIR=/juice5b/scr5b/kaitwang/tmp

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
