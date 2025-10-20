

# sudo su

source /cpfs01/shared/XNLP_H800/miniconda3/bin/activate /cpfs01/shared/XNLP_H800/sunqiushi/conda-envs/ui_agent
cd /cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench

# MODEL_DIR="/cpfs01/shared/XNLP_H800/hf_hub/InternVL3_5-4B-HF"
# MODEL_DIR="/cpfs01/shared/XNLP_H800/hf_hub/InternVL3_5-4B-Instruct"
MODEL_DIR="/cpfs01/shared/XNLP_H800/hf_hub/InternVL3_5-8B-HF"
MODEL_DIR="/cpfs01/shared/XNLP_H800/hf_hub/InternVL3_5-4B-HF"
# MODEL_DIR="/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/model_checkpoints/final0_internvl3.5-4b_0820"


 
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
    --limit-mm-per-prompt.videos 0 \
    --disable-mm-preprocessor-cache \
    --trust-remote-code
