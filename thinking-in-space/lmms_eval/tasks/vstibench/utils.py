import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "obj_obj_relative_pos_nf",
    "obj_obj_relative_pos_ud",
    "obj_obj_relative_pos_lr",
    "camera_obj_rel_dist_v1",
    "camera_obj_rel_dist_v2",
    "camera_obj_rel_dist_v3",
    "camera_movement_direction"
]
NA_QUESTION_TYPES = [
    "camera_obj_abs_dist",
    "camera_displacement",
    "camera_obj_dist_change",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vstibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # # 筛选 camera_movement_direction类型的问题
    # dataset = dataset.filter(lambda x: x['question_type'] in ['camera_movement_direction'])
    
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['mc_answer'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}


def _to_native(v):
    """Convert numpy scalars/arrays to native Python so pd.DataFrame doesn't raise TypeError."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        return v.item()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_native(x) for x in v]
    if isinstance(v, (np.str_, np.bytes_)):
        return str(v)
    return v


def _doc_to_native(doc):
    """Convert numpy scalars in a result doc to native Python so aggregation doesn't raise."""
    return {str(k): _to_native(v) for k, v in doc.items()}


def vsibench_aggregate_results(results):
    """Aggregate without pandas to avoid TypeError from ensure_index on gathered data."""
    normalized = [_doc_to_native(d) for d in results]
    by_type = {}
    for doc in normalized:
        qt = doc.get("question_type")
        if qt is None:
            continue
        if qt not in by_type:
            by_type[qt] = []
        by_type[qt].append(doc)

    output = {}
    for question_type, docs in by_type.items():
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                vals = [d.get(metric) for d in docs if d.get(metric) is not None]
                output[f"{question_type}_{metric}"] = sum(vals) / len(vals) if vals else 0.0
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                vals = [d.get(metric) for d in docs if d.get(metric) is not None]
                output[f"{question_type}_{metric}"] = sum(vals) / len(vals) if vals else 0.0
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    # Full average: mean of all per-sub-type metrics (9 sub-types)
    all_subtype_vals = [v for k, v in output.items() if k != "overall"]
    full_average = sum(all_subtype_vals) / len(all_subtype_vals) if all_subtype_vals else 0.0

    # Paper Table 2: 5 category scores (Cam-Obj Abs., Cam. Displace., Cam. Mov. Dir., Obj-Obj Rel. Pos., Cam-Obj Rel. Dist.)
    cat_cam_abs = output.get("camera_obj_abs_dist_MRA:.5:.95:.05")
    cat_cam_disp = output.get("camera_displacement_MRA:.5:.95:.05")
    cat_cam_mov = output.get("camera_movement_direction_accuracy")
    obj_obj_vals = [output.get(f"obj_obj_relative_pos_{x}_accuracy") for x in ("nf", "ud", "lr")]
    obj_obj_present = [v for v in obj_obj_vals if v is not None]
    cat_obj_obj = sum(obj_obj_present) / len(obj_obj_present) if obj_obj_present else None
    cam_rel_vals = [output.get(f"camera_obj_rel_dist_v{x}_accuracy") for x in (1, 2, 3)]
    cam_rel_present = [v for v in cam_rel_vals if v is not None]
    cat_cam_obj_rel = sum(cam_rel_present) / len(cam_rel_present) if cam_rel_present else None

    category_scores = [
        ("Cam-Obj Abs. Dist.", cat_cam_abs),
        ("Cam. Displace.", cat_cam_disp),
        ("Cam. Mov. Dir.", cat_cam_mov),
        ("Obj-Obj Rel. Pos.", cat_obj_obj),
        ("Cam-Obj Rel. Dist.", cat_cam_obj_rel),
    ]
    five_cats = [v for _, v in category_scores if v is not None]
    paper_average = sum(five_cats) / len(five_cats) if five_cats else 0.0

    eval_logger.info("VSTiBench category scores (paper Table 2):")
    for name, val in category_scores:
        pct = val * 100.0 if val is not None else None
        eval_logger.info(f"  {name}: {pct:.2f}%" if pct is not None else f"  {name}: N/A")
    eval_logger.info(f"Full average (all sub-types): {full_average * 100.0:.2f}%")
    eval_logger.info(f"Paper average (5 categories): {paper_average * 100.0:.2f}%")
    eval_logger.info(f"Evaluation results (all keys): {output}")

    output["overall"] = paper_average
    output["overall_full"] = full_average
    return output["overall"] * 100.0
