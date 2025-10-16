#!/bin/bash
# Script to convert HuggingFace model to GGUF format
# Usage: ./convert_to_gguf.sh

set -e  # Exit on error

echo "Converting merged_model to GGUF format..."

# Check if llama.cpp exists
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp
fi

# Build llama.cpp if not already built
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama.cpp..."
    cd llama.cpp
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build . --config Release -j$(nproc)
    cd ../..
else
    echo "llama.cpp already built, skipping build..."
fi

# Install Python requirements
echo "Installing Python requirements..."
pip install -q -r llama.cpp/requirements.txt

# Create output directory if it doesn't exist
mkdir -p gguf_model

# Step 1: Convert model to GGUF (f16 format)
echo "Step 1: Converting model to GGUF (f16)..."
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile gguf_model/model-f16.gguf \
    --outtype f16

# Step 2: Quantize to q4_k_m
echo "Step 2: Quantizing to q4_k_m..."
./llama.cpp/build/bin/llama-quantize gguf_model/model-f16.gguf gguf_model/model.gguf q4_k_m

# Clean up intermediate file
echo "Cleaning up intermediate files..."
rm gguf_model/model-f16.gguf

echo "✓ Conversion complete! Model saved to gguf_model/model.gguf"
