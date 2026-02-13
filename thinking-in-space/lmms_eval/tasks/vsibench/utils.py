import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
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
    # # 过滤出route_planning类型的数据
    # dataset = dataset.filter(lambda x: x['question_type'] == 'route_planning')
    
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
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
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

    object_rel_direction_tasks = [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
    ]
    direction_accuracies = []
    for task in object_rel_direction_tasks:
        accuracy_key = f"{task}_accuracy"
        if accuracy_key in output:
            direction_accuracies.append(output.pop(accuracy_key))
    if direction_accuracies:
        output["object_rel_direction_accuracy"] = sum(direction_accuracies) / len(direction_accuracies)

    # Table 1: 8 categories (Obj. Count, Abs. Dist., Obj. Size, Room Size, Rel. Dist., Rel. Dir., Route Plan, Appr. Order)
    category_scores = [
        ("Obj. Count", output.get("object_counting_MRA:.5:.95:.05")),
        ("Abs. Dist.", output.get("object_abs_distance_MRA:.5:.95:.05")),
        ("Obj. Size", output.get("object_size_estimation_MRA:.5:.95:.05")),
        ("Room Size", output.get("room_size_estimation_MRA:.5:.95:.05")),
        ("Rel. Dist.", output.get("object_rel_distance_accuracy")),
        ("Rel. Dir.", output.get("object_rel_direction_accuracy")),
        ("Route Plan", output.get("route_planning_accuracy")),
        ("Appr. Order", output.get("obj_appearance_order_accuracy")),
    ]
    eight_cats = [v for _, v in category_scores if v is not None]
    paper_avg = sum(eight_cats) / len(eight_cats) if eight_cats else 0.0

    eval_logger.info("VSI-Bench category scores (paper Table 1):")
    for name, val in category_scores:
        pct = val * 100.0 if val is not None else None
        eval_logger.info(f"  {name}: {pct:.2f}%" if pct is not None else f"  {name}: N/A")
    eval_logger.info(f"Average (8 categories): {paper_avg * 100.0:.2f}%")
    eval_logger.info(f"Evaluation results (all keys): {output}")

    output["overall"] = paper_avg if eight_cats else (sum(output.values()) / len(output) if output else 0.0)
    return output["overall"] * 100.0
