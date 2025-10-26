source activate Path_to_Your_Conda_Environment
cd Project_Directory_of_DesignBench
MODEL_DIR="Specify the model dir here, e.g., XXX/Qwen-2.5-VL-7B"

HOST_IP=$(hostname -i)
model_name=$(basename $MODEL_DIR)
echo "API base_url=http://${HOST_IP}:8994/v1"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_DIR} --served-model-name ${model_name} \
    --port 8994 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 --max-seq-len 32768 \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt.videos 0 \
    --disable-mm-preprocessor-cache \
    --trust-remote-code
