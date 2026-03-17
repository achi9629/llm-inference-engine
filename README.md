# These are the readme section for this project

- Inspired by vLLM architecture
- Overview
- Architecture (architecture.md)
- Features (features.md)
- Benchmarks (benchmarks.md)
- How to run
- Future works

## A lightweight LLM inference engine implementing

- Continuous batching
- Paged KV cache
- Request scheduling
- Token streaming
- Transformer inference

## How to run (Linux)

### 1) Create and activate virtual environment

````bash
python3 -m venv .venv
source .venv/bin/activate
````

### 2) Install dependencies

````bash
python -m pip install --upgrade pip
pip install -r requirements.txt
````

### 3) Verify installation

````bash
python -c "import torch, transformers, fastapi; print('OK', torch.__version__, transformers.__version__)"
````
