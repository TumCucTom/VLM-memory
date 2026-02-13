### Installation

```bash
cd thinking-in-space

conda create --name vsibench python=3.10
conda activate vsibench

# PyTorch with CUDA: use pip (conda pytorch-cuda is not available on linux-aarch64)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
# FlashAttention: x86_64 only; on aarch64 skip or install without it (model may fall back to SDPA)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.40.0 peft==0.10.0 google-generativeai google-genai huggingface_hub[hf_xet]
```

**Note:** On **linux-aarch64** (ARM), `pytorch-cuda=12.1` is not in conda; use the pip line above. If `cu121` wheels are not available for your platform, try `cu118` or omit `--index-url` to use default PyTorch. The FlashAttention wheel is for x86_64; on aarch64 either skip it (PyTorch will use SDPA/eager attention) or build from source.

### 

### Evaluation

We provide an evaluation scripts. You can simply run the following code to start your evaluation.

```bash
bash eval_vlm_3r.sh
```

