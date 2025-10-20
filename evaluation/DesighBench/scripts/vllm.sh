# sudo su
# source /cpfs01/shared/XNLP_H800/miniconda3/bin/activate /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/conda-envs/mclm-eval
source /cpfs01/shared/XNLP_H800/miniconda3/bin/activate /cpfs01/shared/XNLP_H800/sunqiushi/conda-envs/ui_agent
cd /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench
MODEL_DIR="/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/model_checkpoints/final0_qwen2.5-vl-7b_ablation-reward"

 
HOST_IP=$(hostname -i)
model_name=$(basename $MODEL_DIR)
echo "API base_url=http://${HOST_IP}:8994/v1"

export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_DIR} --served-model-name ${model_name} \
    --port 8994 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 --max-seq-len 32768 \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt.images 5 \
    --trust-remote-code

# export VLLM_USE_V1=1
# python3 -m vllm.entrypoints.openai.api_server \
#     --model ${MODEL_DIR} --served-model-name ${model_name} \
#     --port 8994 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768 --max-seq-len 32768 \
#     --tensor-parallel-size 4 \
#     --mm-processor-kwargs.image_size 448 \
#     --mm-processor-kwargs.video_size 448 \
#     --limit-mm-per-prompt.images 8 \
#     --limit-mm-per-prompt.videos 0 \
#     --disable-mm-preprocessor-cache
#     --limit-mm-per-prompt "image=5"


# llama系列要用下面的方式, 添加max_num_seqs这个参数
# python3 -m vllm.entrypoints.openai.api_server \
#     --model ${MODEL_DIR} --served-model-name ${model_name} \
#     --port 8994 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 80000 --max-seq-len 80000 \
#     --tensor-parallel-size 2 \
#     --limit-mm-per-prompt "image=5" \
#     --max_num_seqs 2
