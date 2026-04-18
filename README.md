# Rick LLM 2026 — Fine-tune Qwen3 on Rick & Morty Transcripts

```
Transcripts → (optional Groq-enriched) ShareGPT Dataset → Qwen3-8B (QLoRA) → GGUF → Ollama
```

---

## VRAM Requirements

| Model | VRAM |
|-------|------|
| `unsloth/Qwen3-4B` | ~8GB |
| `unsloth/Qwen3-8B` | ~10GB (default) |
| `unsloth/Qwen3-14B` | ~18GB |

---

## Setup

**Python dependencies:**
```bat
:: Stage 1 (CPU fine)
pip install datasets huggingface_hub pandas groq

:: Stage 2 (GPU required — NVIDIA only)
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes

:: Stage 3 (CPU fine)
pip install huggingface_hub
```

**Set environment variables (Command Prompt):**
```bat
set HF_TOKEN=hf_your_token_here
set GROQ_API_KEY=gsk_your_key_here
```

**Set environment variables (PowerShell):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
$env:GROQ_API_KEY="gsk_your_key_here"
```

**Install Ollama on Windows:**
Download the installer from https://ollama.com/download and run it.

---

## Stage 1: Generate Dataset

1. Download transcripts from Kaggle and place the CSV at `.\data\RickAndMortyScripts.csv`
   https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts

2. Update `HF_DATASET_REPO` in `stage1_dataset.py`

3. Run:
```bat
python stage1_dataset.py
```

`GROQ_API_KEY` is optional — if set, Groq rewrites Rick's lines to be more expressive before pushing to HF. Skip it to use raw transcript pairs.

---

## Stage 2: Finetune

Update `HF_DATASET_REPO` and `HF_MODEL_REPO` in `stage2_finetune.py`, then:
```bat
python stage2_finetune.py
```

Requires an NVIDIA GPU with CUDA. WSL2 is recommended if you run into bitsandbytes issues on Windows native.

---

## Stage 3: Deploy

```bat
python stage3_deploy.py
```

Or directly with Ollama:
```bat
ollama create rick-llm -f .\ollama_files\Modelfile
ollama run rick-llm
```

---

## Troubleshooting

**bitsandbytes not working on Windows:**
Run inside WSL2, or install the Windows-compatible build:
```bat
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

**CUDA out of memory:**
In `stage2_finetune.py`, reduce batch size or switch to a smaller model:
```python
BASE_MODEL = "unsloth/Qwen3-4B"
# and/or in SFTConfig:
per_device_train_batch_size = 1
max_seq_length = 1024
```

**Wrong column names in CSV:**
Check your CSV headers and update the `char_col` / `line_col` detection in `stage1_dataset.py`.
