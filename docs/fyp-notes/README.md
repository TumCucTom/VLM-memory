# Running demo

Created a slurm script at /scripts/video/demo/video_demo.slurm to run the demo script
- This needs larger GPU then just 2080ti - using A100
- Setup and load python env
- Have to use scratch work space
- Running into tokenizer vocab mismatch error
    - Made change to Builder.py 
    - Use actual size rather than hardcoded 152064

- `CUT3R/src/croco/models/pos_embed.py` - Fixed CUDA indexing error in RoPE2D (position clamping)
- `llava/model/language_model/llava_qwen.py` - Fixed MultiheadAttention initialisation

# Added working memory
Run inference by using the demo script to do inital test

Created scripts: 
- `llava/model/memory/working_memory.py` - FIFO working memory module (L_w=8 capacity)

Updated scripts: 
- `llava/model/llava_arch.py` - Integrated working memory with attention retrieval and per-frame updates
- `playground/demo/video_demo.py` - Added memory clearing at start of each video
- `llava/model/language_model/llava_qwen.py` - added debug prints
- `llava/model/builder.py` - Added debug prints for model loading