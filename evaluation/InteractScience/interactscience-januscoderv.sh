#!/bin/bash

# Port for vLLM server
PORT=8000

echo "==== Starting ===="
model_path="model_path"
# Start vLLM API server in the background
vllm serve "$model_path" \
    --port $PORT \
    --tensor-parallel-size 8 \
    --max-model-len 32000 &

SERVER_PID=$!

# # Wait for the server to be ready
echo "Waiting for the server to start..."
until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 1
done
echo "Server is up. Running the test script..."

# Run the test script
echo "Running tests for model: $MODEL"

python test_llm.py \
    --dataset_path "data/interactscience.jsonl" \
    --prompt_type "vlm_system_prompt" \
    --dump_path "eval/interactscience_${MODEL}.jsonl" \
    --model_path "$model_path" \
    --base_url "http://localhost:$PORT/v1" \
    --api_key "EMPTY"

echo "Test completed. Shutting down the server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped."

echo "==============================="
echo