"""Load and validate the final demo TOML without importing GPU libraries."""
import math
import tomllib
from pathlib import Path


def load_final_config(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    config = tomllib.loads(text)
    validate_final_config(config)
    return config


def validate_final_config(config):
    def require(condition, message):
        if not condition:
            raise ValueError("Final config: " + message)

    require(config.get("config_version") == 1, "config_version=1 gerekli.")
    for section in ("model", "checkpoint", "selector", "fusion", "seeds", "evaluation"):
        require(isinstance(config.get(section), dict), f"{section} bölümü eksik.")
    fixed = {
        "model": {"input_size": 224, "input_resize": "bicubic", "num_classes": 21,
                  "normalization": "processor.image_mean_and_image_std",
                  "backbone_frozen": True, "inference_mode": "eval"},
        "selector": {"padding": 0, "entropy": "normalized_softmax_class_entropy",
                     "score": "mean_crop_entropy", "selection": "descending_entropy_greedy_iou",
                     "coordinate_mapping": "floor_start_ceil_end_clip_to_image",
                     "crop_source": "original_rgb_image", "empty_selection": "keep_global_baseline"},
        "fusion": {"mode": "mean_local_logits_global_outside", "logit_resize": "bilinear",
                   "align_corners": False, "prediction": "argmax_after_fusion"},
        "evaluation": {"resolution": "native", "ignore_index": 255, "include_background": True},
    }
    for section, fields in fixed.items():
        for key, expected in fields.items():
            require(config[section].get(key) == expected,
                    f"{section}.{key} için desteklenen değer: {expected!r}")
    for key in ("id", "revision"):
        require(isinstance(config["model"].get(key), str) and bool(config["model"][key]),
                f"model.{key} boş olamaz.")
    selector = config["selector"]
    for key in ("k", "crop_size", "stride"):
        value = selector.get(key)
        require(type(value) is int and value > 0, f"selector.{key} pozitif tamsayı olmalı.")
    require(selector["crop_size"] <= 224, "crop_size 224'ten büyük olamaz.")
    for key in ("minimum_entropy", "overlap_iou"):
        value = selector.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                f"selector.{key} [0,1] aralığında olmalı.")
    require(type(config["seeds"].get("base")) is int and 0 <= config["seeds"]["base"] < 2**32,
            "seeds.base 32-bit negatif olmayan tamsayı olmalı.")
    digest = config["checkpoint"].get("sha256", "")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "checkpoint.sha256 geçersiz.")
    require(config["checkpoint"].get("head_type") == "conv1x1", "head_type conv1x1 olmalı.")
    require(type(config["checkpoint"].get("epoch")) is int, "checkpoint.epoch tamsayı olmalı.")
