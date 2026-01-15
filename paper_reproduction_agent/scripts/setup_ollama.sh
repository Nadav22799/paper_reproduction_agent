#!/bin/bash
# Ollama Setup Script for GPT-OSS
# This script will install Ollama and set up GPT-OSS model

set -e  # Exit on error

echo "=========================================="
echo "📦 Ollama Setup for GPT-OSS"
echo "=========================================="

# Step 1: Install Ollama
echo ""
echo "Step 1: Installing Ollama..."
echo "=========================================="
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
    ollama --version
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed successfully"
fi

# Step 2: Start Ollama service
echo ""
echo "Step 2: Starting Ollama service..."
echo "=========================================="
# Check if ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service is already running"
else
    echo "Starting Ollama service..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    echo "✅ Ollama service started"
fi

# Step 3: Pull GPT-OSS model
echo ""
echo "Step 3: Pulling GPT-OSS 20B model..."
echo "=========================================="
echo "⚠️  This will download ~12GB and requires at least 16GB VRAM"
echo "This may take 5-15 minutes depending on your internet speed..."
ollama pull gpt-oss:20b

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
echo "=========================================="
echo "Available models:"
ollama list

echo ""
echo "Testing Ollama API:"
curl -s http://localhost:11434/api/tags | head -20

echo ""
echo "=========================================="
echo "✅ Ollama Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update your .env file (I'll do this for you)"
echo "2. Install Python dependencies: pip install langchain-ollama"
echo "3. Run test: python test_vllm_fix.py"
echo ""
