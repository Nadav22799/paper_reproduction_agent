#!/bin/bash
# vLLM Server Startup Script
# This runs vLLM as an OpenAI-compatible API server

# Configuration
MODEL="openai/gpt-oss-20b"
HOST="0.0.0.0"
PORT=8000
DTYPE="bfloat16"  # GPT-OSS uses bf16 instead of fp16
MAX_MODEL_LEN=4096  # Increased context length for larger model
GPU_MEMORY_UTILIZATION=0.9  # Increased for 20B model

echo "=========================================="
echo "🚀 Starting vLLM Server"
echo "=========================================="
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Data Type: $DTYPE"
echo "Max Context Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "=========================================="
echo ""
echo "Server will be available at: http://localhost:${PORT}/v1"
echo "OpenAI-compatible endpoint: http://localhost:${PORT}/v1/chat/completions"
echo ""
echo "To stop the server: Press Ctrl+C"
echo ""

# Check if HuggingFace token is set
if [ -z "$HUGGINGFACE_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: No HuggingFace token found in environment"
    echo "If the model is gated, you may need to set HUGGINGFACE_TOKEN or HF_TOKEN"
    echo ""
fi

# Start vLLM server
vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser "hermes"
