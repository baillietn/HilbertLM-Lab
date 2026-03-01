# 🤖 NanoLLM

A **135M parameter language model** trained from scratch with PyTorch on 20 billion tokens

---

## What is NanoLLM?

NanoLLM is a complete educational project demonstrating how to build a modern transformer-based language model from scratch, including:

- Custom BPE tokenizer (49,152 vocab)
- Modern architecture (GQA, RoPE, SwiGLU)
- Curriculum learning across 4 stages
- Supervised fine-tuning for chat
- Loss landscape visualization
- Interactive web demo

**Home Page:** [NanoLLM](https://nico77310.github.io/NanoLLM/)
**Chat Page:** [NanoLLM Chat](https://nico77310.github.io/NanoLLM/chat/)

---

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.8+ (for GPU training)
- ~12GB GPU VRAM (RTX 3080/4080/5080 or similar)

### Setup

```bash
# Clone the repository
git clone https://github.com/Nico77310/NanoLLM.git
cd NanoLLM

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install tokenizers numpy tqdm prompt_toolkit datasets

# Optional: Transformer Engine for FP8 (Ada/Hopper/Blackwell GPUs)
pip install transformer-engine[pytorch]
```

---

## Quick Start

### 1. Download & Prepare Data

```bash
# Download datasets from HuggingFace (FineWeb-Edu, Python-Edu, FineMath, Cosmopedia, SmolTalk)
python src/get_data.py

# Prepare SFT dataset
python src/get_data.py --prepare-sft

# Generate validation batches (optional, for loss landscape)
python src/get_data.py --validation
```

### 2. Train the Base Model

```bash
# Pre-training 
python src/train.py --precision bf16 --compile-mode default

# Resume from checkpoint
python src/train.py --precision bf16 --checkpoint checkpoints/checkpoint.pt
```

**Configuration:** Edit [`src/config.py`](src/config.py) to adjust hyperparameters (batch size, learning rate, model dimensions, precision).

### 3. Supervised Fine-Tuning

```bash
# Fine-tune on conversational data (100M tokens)
python src/train.py --sft --precision bf16 --checkpoint checkpoints/final_base_model.pt
```

### 4. Generate Text

```bash
# Interactive chat
python src/generate.py --ckpt checkpoints/nano_llm_chat_final.pt

# Base model (no chat formatting)
python src/generate.py --ckpt checkpoints/final_base_model.pt --base
```

### 5. Visualize Training

```bash
# Plot training metrics
python src/plot_metrics.py --pretrain checkpoints/final_base_model.pt --sft checkpoints/nano_llm_chat_final.pt

# Generate loss landscape animation (requires validation batches)
python src/plot_landscape.py --checkpoint checkpoints/final_base_model.pt
```

---
## Training Options

```bash
# Mixed precision
python src/train.py --precision bf16        # BFloat16 (default)
python src/train.py --precision fp8         # FP8 (Transformer Engine)

# Torch compile modes
python src/train.py --compile-mode default  # Balanced (default)
python src/train.py --compile-mode reduce-overhead
python src/train.py --compile-mode max-autotune
python src/train.py --compile-mode none     # Disable

# Micro-batch size (for gradient accumulation)
python src/train.py --micro-batch-size 4    # Effective batch = batch_size / micro_batch_size
```
---

## Key Features

| Feature | Description |
|---------|-------------|
| **Architecture** | Grouped Query Attention (GQA), RoPE, SwiGLU, LayerNorm |
| **Training** | Curriculum learning (4 stages), BF16/FP8, torch.compile |
| **Data** | 20B tokens (FineWeb-Edu, Python-Edu, FineMath, Cosmopedia) |
| **SFT** | 100M conversational tokens (SmolTalk) |
---

## Documentation

- **Technical Report:** [NanoLLM](https://nico77310.github.io/NanoLLM/) — Complete architecture, training methodology, loss landscape analysis
- **Config File:** [`src/config.py`](src/config.py) — Adjust hyperparameters here


