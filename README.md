## Project Overview

This repository contains a workflow for fine-tuning language models using Unsloth and converting them to GGUF format for deployment with Ollama. The primary use case is fine-tuning smaller language models (like Phi-3) for structured JSON extraction from HTML content.

## Key Components

### 1. Fine-tuning Pipeline (test.ipynb)
The notebook implements the complete fine-tuning workflow:
- Uses Unsloth library for fast, memory-efficient fine-tuning
- Base model: `unsloth/Phi-3-mini-4k-instruct-bnb-4bit` (4-bit quantized)
- Training approach: LoRA (Low-Rank Adaptation) with rank=64
- Dataset: JSON extraction examples (500 samples) in format `{"input": "...", "output": {...}}`
- Training configuration: 3 epochs, batch size 2, gradient accumulation 4
- Output: Saved checkpoints in `outputs/` directory

### 2. Model Export Pipeline
After fine-tuning, the workflow supports multiple export paths:

**Merge LoRA Adapters:**
- Loads the latest checkpoint from `outputs/checkpoint-*`
- Merges LoRA adapters back into base model
- Saves merged model to `merged_model/` directory

**Convert to GGUF:** (via `convert_to_gguf.sh`)
- Uses llama.cpp for conversion
- Converts merged HuggingFace model → GGUF f16 → quantized q4_k_m
- Final output: `gguf_model/model.gguf`

**Deploy with Ollama:** (via `Modelfile`)
- Creates an Ollama model from the GGUF file
- Includes custom template for Phi-3 chat format
- System prompt configured for JSON extraction tasks

## Directory Structure

```
finetuning/
├── test.ipynb                   # Main fine-tuning notebook
├── json_extraction_dataset_500.json  # Training dataset
├── convert_to_gguf.sh          # Conversion script (HF → GGUF)
├── Modelfile                   # Ollama model configuration
├── outputs/                    # Training checkpoints
│   └── checkpoint-*/
├── merged_model/              # Merged LoRA + base model
├── gguf_model/                # GGUF format model
└── llama.cpp/                 # Cloned conversion tools (auto-generated)
```

## Common Commands

### Fine-tuning
Open and run cells in `test.ipynb` sequentially. Key steps:
1. Load base model with Unsloth
2. Prepare dataset from JSON file
3. Add LoRA adapters
4. Train with SFTTrainer
5. Test the fine-tuned model
6. Merge LoRA adapters (optional, for export)

### Model Conversion
```bash
# Convert merged model to GGUF format
./convert_to_gguf.sh
```

This script will:
- Clone llama.cpp if needed
- Build llama.cpp tools
- Convert HuggingFace model to GGUF f16
- Quantize to q4_k_m format

### Ollama Deployment
```bash
# Create Ollama model from GGUF
ollama create my-finetuned-model -f Modelfile

# Test the model
ollama run my-finetuned-model "Extract product info from: <div>...</div>"
```

## Important Notes

### GPU Requirements
- This workflow requires CUDA-capable GPU (tested with RTX 4090)
- Unsloth enables 4-bit quantized training to reduce memory usage
- Training uses mixed precision (bf16 if supported, else fp16)

### Model Architecture
- LoRA targets all attention and MLP layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- LoRA configuration: rank=64, alpha=128, dropout=0
- Uses Unsloth's gradient checkpointing for memory efficiency

### Dataset Format
Training data must follow this structure:
```json
{
  "input": "instruction text with data to extract",
  "output": {"key": "value", ...}
}
```

The format_prompt function converts this to:
```
### Input: {input}
### Output: {json.dumps(output)}<|endoftext|>
```

### Checkpoint Management
- Checkpoints saved every epoch to `outputs/checkpoint-*`
- Only last 2 checkpoints kept (save_total_limit=2)
- Load latest checkpoint by sorting checkpoint numbers numerically

### llama.cpp Integration
- The conversion script clones and builds llama.cpp automatically
- Build configuration: Release mode, no CURL support
- Uses `convert_hf_to_gguf.py` for initial conversion
- Quantization uses `llama-quantize` binary for q4_k_m format
- 
## Run in terminal:
 ollama create my-finetuned-model -f Modelfile

## Test it in ollama:
 ollama run my-finetuned-model "Extract product info from: <div>...</div>"

