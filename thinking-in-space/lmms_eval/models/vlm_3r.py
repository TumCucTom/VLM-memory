import copy
import gc
import json
import math
import os
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

import sys; sys.path = ["../../VLM-memory/"] + sys.path
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError:
    eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

try:
    from llava.model.language_model.llava_qwen import LlavaQwenConfig

    AutoConfig.register("llava_qwen", LlavaQwenConfig)
except ImportError:
    eval_logger.debug("No Qwen for llava vid")

from llava.model.language_model.llava_llama import LlavaConfig

AutoConfig.register("llava_llama", LlavaConfig)


@register_model("vlm_3r")
class Vlm3r(lmms):
    """
    Vlm3r Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/VLM-3R-7B-Qwen2",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        max_frames_num: int = 3,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "bilinear",
        mm_newline_position: str = "grid",
        mm_pooling_position: str = "after",
        overwrite: bool = True,
        video_decode_backend: str = "pyav",
        delay_load: bool = False,
        tie_weights: bool = True,
        model_name: str = None,
        model_base: str = None,
        use_dual_memory: bool = True,
        memory_mode: str = "working_only",  # "working_only", "episodic_only", "both", or "off"
        memory_alpha: float = 0.3,  # fraction of output that comes from memory (0.3 = 30% memory, 70% original)
        use_query_selection: Optional[bool] = None,
        query_selection_num_select: Optional[int] = None,
        query_selection_temperature: Optional[float] = None,
        query_selection_use_gumbel: Optional[bool] = None,
        query_selection_project_selected: Optional[bool] = None,
        query_selection_trace_path: Optional[str] = None,
        query_selection_trace_limit: Optional[int] = None,
        frames_upbound: Optional[int] = None,
        force_sample: Optional[bool] = None,
        checkpoint_adapter: str = None,  # optional: load base+LoRA then overlay adapter from this path (for pipeline verification)
        overlay_memory_only: bool = False,  # if True, only overlay working/episodic memory weights (fusion_block etc. are frozen in training so should match base; use as safeguard if eval load path differs)
        **kwargs,
    ) -> None:
        super().__init__()

        torch.backends.cudnn.enabled = False

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = get_model_name_from_path(pretrained)
        # Local full checkpoint (work_dirs/...[/checkpoint-*]): use repo llava and ensure builder picks LLaVA-Qwen.
        _load_pretrained_model = load_pretrained_model
        if "work_dirs" in str(pretrained):
            import os as _os
            import importlib
            _repo = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../.."))
            if _repo not in sys.path:
                sys.path.insert(0, _repo)
            if self.model_name and "llava" not in self.model_name.lower():
                self.model_name = "llava_qwen"
            import llava.model.builder as _builder_module
            importlib.reload(_builder_module)
            _load_pretrained_model = _builder_module.load_pretrained_model
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        self.mm_resampler_location = mm_pooling_position
        self.mm_newline_position = mm_newline_position
        self.delay_load = delay_load
        self.query_selection_trace_path = query_selection_trace_path
        self.query_selection_trace_limit = int(query_selection_trace_limit) if query_selection_trace_limit is not None else None
        self._query_selection_trace_count = 0

        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = self.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_pooling_position"] = self.mm_resampler_location
            overwrite_config["mm_newline_position"] = self.mm_newline_position
            overwrite_config["add_faster_video"] = False
            overwrite_config["delay_load"] = self.delay_load
            overwrite_config["use_dual_memory"] = use_dual_memory
            overwrite_config["memory_mode"] = memory_mode
            overwrite_config["memory_alpha"] = memory_alpha
            overwrite_config["attn_implementation"] = attn_implementation

            def _as_bool(value):
                if isinstance(value, str):
                    return value.lower() in ("1", "true", "yes", "y")
                return bool(value)

            if use_query_selection is not None:
                overwrite_config["use_query_selection"] = _as_bool(use_query_selection)
            if query_selection_num_select is not None:
                overwrite_config["query_selection_num_select"] = int(query_selection_num_select)
            if query_selection_temperature is not None:
                overwrite_config["query_selection_temperature"] = float(query_selection_temperature)
            if query_selection_use_gumbel is not None:
                overwrite_config["query_selection_use_gumbel"] = _as_bool(query_selection_use_gumbel)
            if query_selection_project_selected is not None:
                overwrite_config["query_selection_project_selected"] = _as_bool(query_selection_project_selected)
            if frames_upbound is not None:
                overwrite_config["frames_upbound"] = int(frames_upbound)
            if force_sample is not None:
                overwrite_config["force_sample"] = _as_bool(force_sample)

            # When using checkpoint_adapter, merge memory config from checkpoint so model is built with memory modules (good base + overlay).
            # Spatial/fusion (spatial_tower, fusion_block, etc.) stay from HF pretrained — just in case.
            if checkpoint_adapter and model_base and os.path.isdir(checkpoint_adapter):
                _ckpt_config_path = os.path.join(checkpoint_adapter, "config.json")
                if os.path.isfile(_ckpt_config_path):
                    import json
                    with open(_ckpt_config_path) as _f:
                        _ckpt_cfg = json.load(_f)
                    for _k in (
                        "working_memory_size",
                        "episodic_memory_size",
                        "episodic_memory_gated_attention",
                        "memory_fusion_hidden_dim",
                        "memory_mode",
                        "use_query_selection",
                        "query_selection_num_select",
                        "query_selection_temperature",
                        "query_selection_use_gumbel",
                        "query_selection_project_selected",
                        "frames_upbound",
                        "force_sample",
                    ):
                        if _k in _ckpt_cfg:
                            overwrite_config[_k] = _ckpt_cfg[_k]
                    eval_logger.info(f"Merged memory config from checkpoint: {[k for k in overwrite_config if k in ('working_memory_size', 'episodic_memory_size', 'episodic_memory_gated_attention', 'memory_fusion_hidden_dim', 'memory_mode')]}")

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

            if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
                if "224" in cfg_pretrained.mm_vision_tower:
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            if "v1.5" in pretrained:  # A hardcode solution here to load v1.5 model, otherwise it will use LlavaConfig from hf transformers
                from llavavid.model.language_model.llava_llama import (
                    LlavaConfig,
                    LlavaLlamaForCausalLM,
                )
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
                cfg_pretrained = LlavaConfig.from_pretrained(pretrained)
                if overwrite_config is not None:
                    eval_logger.log(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                kwargs["torch_dtype"] = torch.float16
                self._model = LlavaLlamaForCausalLM.from_pretrained(pretrained, low_cpu_mem_usage=True, config=cfg_pretrained, device_map=self.device_map, **kwargs)
                vision_tower = self._model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model(device_map=self.device_map)
                if self.device_map != "auto":
                    vision_tower.to(device="cuda", dtype=torch.float16)
                self._image_processor = vision_tower.image_processor

                if hasattr(self._model.config, "max_sequence_length"):
                    self._max_length = self._model.config.max_sequence_length
                else:
                    self._max_length = 2048
            else:
                self._tokenizer, self._model, self._image_processor, self._max_length = _load_pretrained_model(pretrained, model_base, self.model_name, device_map=self.device_map, overwrite_config=overwrite_config)
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = _load_pretrained_model(
                pretrained,
                None,
                self.model_name,
                device_map=self.device_map,
            )

        # Overlay adapter + memory weights from checkpoint (good base from base+LoRA, then overlay trained adapter/memory)
        if checkpoint_adapter and model_base and os.path.isdir(checkpoint_adapter):
            from safetensors.torch import load_file
            # State_dict keys use one "model." prefix for LLaVA parts (LlavaQwenForCausalLM.model = LlavaQwenModel).
            # "model.model." is the inner Qwen2Model; memory/projectors live under "model.*".
            _all_prefixes = (
                "model.fusion_block.",
                "model.mm_projector.",
                "model.vision_resampler.",
                "model.working_memory.",
                "model.episodic_memory.",
                "model.memory_fusion_mlp.",
                "model.spatial_tower.",
                "model.working_attention.",
                "model.episodic_attention.",
                "model.query_selection.",
            )
            _memory_only_prefixes = (
                "model.working_memory.",
                "model.episodic_memory.",
                "model.memory_fusion_mlp.",
                "model.working_attention.",
                "model.episodic_attention.",
            )
            _adapter_prefixes = _memory_only_prefixes if overlay_memory_only else _all_prefixes
            if overlay_memory_only:
                eval_logger.info("Overlaying only memory weights from checkpoint (fusion_block etc. left from pretrained)")
            # 1) Overlay from safetensors (LoRA shards; may also contain extra state_dict if saved that way)
            index_path = os.path.join(checkpoint_adapter, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                import json
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                adapter_keys = [k for k in weight_map if any(k.startswith(p) for p in _adapter_prefixes)]
                if adapter_keys:
                    keys_by_shard = defaultdict(list)
                    for key in adapter_keys:
                        keys_by_shard[weight_map[key]].append(key)
                    overlaid = 0
                    for shard, shard_keys in sorted(keys_by_shard.items()):
                        path = os.path.join(checkpoint_adapter, shard)
                        if os.path.isfile(path):
                            shard_state = load_file(path)
                            adapter_dict = {k: shard_state[k] for k in shard_keys if k in shard_state}
                            if adapter_dict:
                                self._model.load_state_dict(adapter_dict, strict=False)
                                overlaid += len(adapter_dict)
                            del shard_state, adapter_dict
                            gc.collect()
                    if overlaid:
                        eval_logger.info(f"Overlaid {overlaid} adapter weights from safetensors in {checkpoint_adapter}")
            # 2) Overlay non-LoRA trainables (working_memory, episodic_memory, etc.) — same key handling as builder
            non_lora_path = os.path.join(checkpoint_adapter, "non_lora_trainables.bin")
            if os.path.isfile(non_lora_path):
                non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=True)
                non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
                if any(k.startswith("model.model.") for k in non_lora):
                    non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
                if overlay_memory_only:
                    non_lora = {k: v for k, v in non_lora.items() if any(k.startswith(p) for p in _memory_only_prefixes)}
                if non_lora:
                    self._model.load_state_dict(non_lora, strict=False)
                    eval_logger.info(f"Overlaid {len(non_lora)} non-LoRA trainables from {non_lora_path}")

        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        if not hasattr(self, "_world_size"):
            self._world_size = 1
        self._prepare_query_selection_trace()

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def _prepare_query_selection_trace(self):
        if not self.query_selection_trace_path:
            return
        trace_path = os.path.expanduser(str(self.query_selection_trace_path))
        if self.world_size > 1:
            root, ext = os.path.splitext(trace_path)
            trace_path = f"{root}.rank{self.rank}{ext or '.jsonl'}"
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8"):
            pass
        self.query_selection_trace_path = trace_path
        eval_logger.info(f"Writing query-selection traces to {trace_path}")

    def _get_query_selection_module(self):
        model = self.model
        candidates = []

        def add_candidate(obj):
            if obj is not None and all(obj is not existing for existing in candidates):
                candidates.append(obj)

        add_candidate(model)
        if hasattr(model, "base_model"):
            add_candidate(model.base_model)
            add_candidate(getattr(model.base_model, "model", None))
        if hasattr(model, "get_base_model"):
            try:
                base_model = model.get_base_model()
            except Exception:
                base_model = None
            add_candidate(base_model)
            add_candidate(getattr(base_model, "model", None))

        for obj in list(candidates):
            get_model = getattr(obj, "get_model", None)
            if callable(get_model):
                try:
                    add_candidate(get_model())
                except Exception:
                    pass

        for obj in candidates:
            get_qs = getattr(obj, "get_query_selection", None)
            if callable(get_qs):
                try:
                    qs = get_qs()
                except Exception:
                    qs = None
            else:
                qs = getattr(obj, "query_selection", None)
            if qs is not None:
                return qs
        return None

    def _video_sampling_metadata(self, video_path, num_candidates):
        metadata = {
            "candidate_frame_indices": None,
            "candidate_frame_times_s": None,
            "total_video_frames": None,
            "video_fps": None,
        }
        if not video_path or not num_candidates:
            return metadata

        try:
            import av

            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            fps = float(stream.average_rate) if stream.average_rate is not None else None
            if total_frames <= 0:
                total_frames = sum(1 for _ in container.decode(video=0))
            container.close()
            if total_frames <= 0:
                return metadata

            sampled = min(int(total_frames), int(num_candidates))
            indices = np.linspace(0, total_frames - 1, sampled, dtype=int).tolist()
            if len(indices) > int(num_candidates):
                indices = indices[: int(num_candidates)]
            times = [round(float(i) / fps, 4) if fps else None for i in indices]
            metadata.update(
                {
                    "candidate_frame_indices": [int(i) for i in indices],
                    "candidate_frame_times_s": times,
                    "total_video_frames": int(total_frames),
                    "video_fps": float(fps) if fps else None,
                }
            )
        except Exception as exc:
            metadata["sampling_metadata_error"] = str(exc)
        return metadata

    def _write_query_selection_trace(self, *, doc_id, task, split, contexts, visuals, output):
        if not self.query_selection_trace_path:
            return
        if self.query_selection_trace_limit is not None and self._query_selection_trace_count >= self.query_selection_trace_limit:
            return

        qs = self._get_query_selection_module()
        trace = getattr(qs, "last_trace", None) if qs is not None else None
        if not trace:
            return

        def first_batch(value):
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                return value[0]
            return value

        selected_indices = [int(i) for i in first_batch(trace.get("selected_indices", []))]
        selected_scores = [float(x) for x in first_batch(trace.get("selected_scores", []))]
        similarities = [float(x) for x in first_batch(trace.get("similarities", []))]
        num_candidates = int(trace.get("num_candidates") or len(similarities))
        video_path = visuals[0] if visuals else None
        video_meta = self._video_sampling_metadata(video_path, num_candidates)
        candidate_frame_indices = video_meta.get("candidate_frame_indices") or []
        candidate_frame_times_s = video_meta.get("candidate_frame_times_s") or []

        def lookup(items, index):
            return items[index] if 0 <= index < len(items) else None

        selected_original_frame_indices = [lookup(candidate_frame_indices, i) for i in selected_indices]
        selected_frame_times_s = [lookup(candidate_frame_times_s, i) for i in selected_indices]

        try:
            doc = self.task_dict[task][split][doc_id]
        except Exception:
            doc = {}
        doc_summary = {}
        if isinstance(doc, dict):
            for key in (
                "question",
                "video_path",
                "category",
                "question_type",
                "task_type",
                "ground_truth",
                "mc_answer",
                "answer",
            ):
                if key in doc:
                    value = doc[key]
                    if isinstance(value, np.generic):
                        value = value.item()
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        value = str(value)
                    doc_summary[key] = value

        record = {
            "doc_id": int(doc_id),
            "task": task,
            "split": split,
            "rank": int(self.rank),
            "context": contexts,
            "output": output,
            "video_path": video_path,
            "doc": doc_summary,
            "num_candidates": num_candidates,
            "num_selected": int(trace.get("num_selected") or len(selected_indices)),
            "selected_candidate_indices": selected_indices,
            "selected_original_frame_indices": selected_original_frame_indices,
            "selected_frame_times_s": selected_frame_times_s,
            "selected_scores": selected_scores,
            "candidate_scores": similarities,
            "candidate_frame_indices": candidate_frame_indices,
            "candidate_frame_times_s": candidate_frame_times_s,
            "total_video_frames": video_meta.get("total_video_frames"),
            "video_fps": video_meta.get("video_fps"),
            "trace_meta": {
                "used_gumbel": bool(trace.get("used_gumbel", False)),
                "temperature": trace.get("temperature"),
                "project_selected": trace.get("project_selected"),
                "clip_feature_shape": trace.get("clip_feature_shape"),
                "question_embed_shape": trace.get("question_embed_shape"),
            },
        }
        if video_meta.get("sampling_metadata_error"):
            record["sampling_metadata_error"] = video_meta["sampling_metadata_error"]

        with open(self.query_selection_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._query_selection_trace_count += 1

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Clear dual memory before each sample so state does not leak across videos (VLM² per-video memory).
            _model = self.model
            if hasattr(_model, "base_model") and hasattr(_model.base_model, "model"):
                _model = _model.base_model.model
            elif hasattr(_model, "get_base_model"):
                _model = _model.get_base_model()
            if getattr(_model, "clear_all_memory", None) is not None:
                _model.clear_all_memory()

            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                videos = []
                try:
                    for visual in visuals:
                        if self.video_decode_backend == "decord":
                            video = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            video = read_video_pyav(visual, num_frm=self.max_frames_num)
                        # video = self.load_video(visual, self.max_frames_num)
                        video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                        videos.append(video)
                except Exception as e:
                    eval_logger.info(f"{e}")
                    eval_logger.info(f"Video {visuals} can not load, check the source")
                    video_path = "\n".join(visuals)
                    res.append(f"Video {video_path} can not load, check the source")
                    pbar.update(1)
                    continue

                qs = contexts
                if self.model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs
            else:
                videos = None
                qs = contexts

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = contexts

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities=["video" for _ in videos] if videos is not None else None,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # inputs = self.tokenizer.batch_decode(input_ids % self.tokenizer.vocab_size, skip_special_tokens=True)[0].strip()
            # print(inputs, outputs)
            self._write_query_selection_trace(
                doc_id=doc_id,
                task=task,
                split=split,
                contexts=contexts,
                visuals=visuals if visuals != [None] else [],
                output=outputs,
            )
            res.append(outputs)
            pbar.update(1)
        return res
