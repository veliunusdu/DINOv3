"""Colab demo helpers. Execute after loading restored_model and processor."""
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import InterpolationMode


def prepare_input(image):
    resized = TF.resize(image.convert("RGB"), [224, 224],
                        interpolation=InterpolationMode.BICUBIC)
    tensor = TF.normalize(TF.to_tensor(resized),
                          processor.image_mean, processor.image_std)
    return tensor.unsqueeze(0).to(device)


@torch.inference_mode()
def predict_logits(image):
    restored_model.eval()
    return restored_model(prepare_input(image)).float().cpu()


def normalized_entropy(logits):
    probabilities = logits.float().softmax(1)
    return (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
            / math.log(logits.shape[1])).clamp(0, 1)


def confusion_for(prediction, target):
    prediction = torch.as_tensor(prediction, dtype=torch.int64).cpu()
    target = torch.as_tensor(target, dtype=torch.int64).cpu()
    assert prediction.shape == target.shape
    valid = target != 255
    gt, pred = target[valid], prediction[valid]
    assert torch.all((gt >= 0) & (gt < 21))
    assert torch.all((pred >= 0) & (pred < 21))
    return torch.bincount(gt * 21 + pred, minlength=441).reshape(21, 21)


def image_miou(prediction, target):
    matrix = confusion_for(prediction, target).double()
    diagonal = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    present = union > 0
    return (diagonal[present] / union[present]).mean().item() if present.any() else None


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = (max(0, min(ax2, bx2) - max(ax1, bx1))
                    * max(0, min(ay2, by2) - max(ay1, by1)))
    union = ((ax2 - ax1) * (ay2 - ay1)
             + (bx2 - bx1) * (by2 - by1) - intersection)
    return intersection / union if union > 0 else 0.0


def select_entropy_topk(entropy_map, k=3, crop_size=112, stride=28,
                        overlap_threshold=0.30, minimum_entropy=0.0):
    entropy_map = torch.as_tensor(entropy_map).detach().float().cpu()
    assert entropy_map.ndim == 2 and torch.isfinite(entropy_map).all()
    assert isinstance(k, int) and k > 0
    assert isinstance(crop_size, int) and crop_size > 0
    assert isinstance(stride, int) and stride > 0
    assert 0 <= overlap_threshold <= 1
    assert 0 <= minimum_entropy <= 1
    height, width = entropy_map.shape
    assert height > 0 and width > 0
    crop_h, crop_w = min(crop_size, height), min(crop_size, width)

    def positions(length, size):
        starts = list(range(0, length - size + 1, stride))
        if starts[-1] != length - size:
            starts.append(length - size)
        return starts

    candidates = []
    for y in positions(height, crop_h):
        for x in positions(width, crop_w):
            candidates.append({
                "box": [x, y, x + crop_w, y + crop_h],
                "score": entropy_map[y:y + crop_h, x:x + crop_w].mean().item(),
            })
    ranked = sorted(candidates, key=lambda region: region["score"], reverse=True)
    selected = []
    for candidate in ranked:
        if candidate["score"] < minimum_entropy:
            continue
        if all(box_iou(candidate["box"], previous["box"]) <= overlap_threshold
               for previous in selected):
            selected.append({"rank": len(selected) + 1,
                             "box": candidate["box"].copy(),
                             "score": candidate["score"]})
            if len(selected) == k:
                break
    return selected, ranked


def topk_box_to_original(box, image_size, map_size=(224, 224)):
    width, height = image_size
    map_h, map_w = map_size
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= map_w and 0 <= y1 < y2 <= map_h
    result = [max(0, math.floor(x1 * width / map_w)),
              max(0, math.floor(y1 * height / map_h)),
              min(width, math.ceil(x2 * width / map_w)),
              min(height, math.ceil(y2 * height / map_h))]
    assert result[0] < result[2] and result[1] < result[3]
    return result


def run_adaptive_zoom(image, k=3, crop_size=112, stride=28,
                      overlap_threshold=0.30, minimum_entropy=0.0):
    """No GT is accepted: selection and fusion depend only on the image."""
    image = image.convert("RGB")
    width, height = image.size
    low = predict_logits(image)
    global_logits = F.interpolate(low, size=(height, width),
                                  mode="bilinear", align_corners=False)
    entropy = normalized_entropy(low)[0]
    regions, candidates = select_entropy_topk(
        entropy, k, crop_size, stride, overlap_threshold, minimum_entropy)
    local_sum = torch.zeros_like(global_logits)
    coverage = torch.zeros((height, width), dtype=torch.int32)
    local_outputs = []
    for region in regions:
        box = topk_box_to_original(region["box"], image.size, tuple(entropy.shape))
        region["original_box"] = box
        x1, y1, x2, y2 = box
        crop = image.crop(tuple(box))
        crop_low = predict_logits(crop)
        local = F.interpolate(crop_low, size=(y2 - y1, x2 - x1),
                              mode="bilinear", align_corners=False)
        assert torch.isfinite(local).all()
        local_sum[:, :, y1:y2, x1:x2] += local
        coverage[y1:y2, x1:x2] += 1
        local_outputs.append({"crop": crop, "logits_224": crop_low,
                              "prediction": local.argmax(1)[0]})
    covered = coverage > 0
    average = local_sum / coverage.clamp_min(1)[None, None]
    fused = torch.where(covered[None, None], average, global_logits)
    assert torch.isfinite(fused).all()
    assert torch.equal(fused[0, :, ~covered], global_logits[0, :, ~covered])
    return {"baseline": global_logits.argmax(1)[0],
            "final": fused.argmax(1)[0], "global_logits_224": low,
            "entropy": entropy, "regions": regions, "candidates": candidates,
            "locals": local_outputs, "coverage": coverage,
            "parameters": {"k_requested": k, "k_selected": len(regions),
                           "crop_size": crop_size, "stride": stride,
                           "overlap_threshold": overlap_threshold,
                           "minimum_entropy": minimum_entropy, "padding": 0}}


def voc_palette():
    palette = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        value = index
        for bit in range(8):
            for channel in range(3):
                palette[index, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    palette[255] = [160, 160, 160]
    return palette


def colorize(mask):
    array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    return voc_palette()[array.astype(np.uint8)]


def save_and_show_demo(image, result, output_dir, target=None, source_name="image"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    image.save(output_dir / "original.png")
    if target is not None:
        assert tuple(target.shape) == (image.height, image.width)
        Image.fromarray(target.numpy().astype(np.uint8)).save(output_dir / "ground_truth.png")

    for name in ("baseline", "final"):
        labels = result[name].numpy().astype(np.uint8)
        Image.fromarray(labels).save(output_dir / f"{name}_labels.png")
        Image.fromarray(colorize(labels)).save(output_dir / f"{name}_color.png")

    np.savez_compressed(output_dir / "global_output.npz",
                        logits_224=result["global_logits_224"].numpy(),
                        entropy=result["entropy"].numpy(),
                        coverage=result["coverage"].numpy())

    tiles = []
    for region, local in zip(result["regions"], result["locals"]):
        rank = region["rank"]
        local["crop"].save(output_dir / f"crop_{rank:02d}.png")
        np.savez_compressed(output_dir / f"crop_{rank:02d}_output.npz",
                            logits_224=local["logits_224"].numpy(),
                            prediction_native=local["prediction"].numpy().astype(np.uint8),
                            original_box=np.array(region["original_box"]))
        tile = Image.new("RGB", (224, 248), "white")
        colored = Image.fromarray(colorize(local["prediction"]))
        thumbnail = ImageOps.contain(colored, (224, 224), Image.Resampling.NEAREST)
        tile.paste(thumbnail, ((224 - thumbnail.width) // 2, 24))
        ImageDraw.Draw(tile).text((6, 5), f"Crop {rank}", fill="black")
        tiles.append(tile)

    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    montage = Image.new("RGB", (224 * columns, 248 * rows), "white")
    for i, tile in enumerate(tiles):
        montage.paste(tile, ((i % columns) * 224, (i // columns) * 248))
    montage.save(output_dir / "local_predictions.png")

    metrics = {}
    for name in ("baseline", "final"):
        if target is not None:
            valid = target != 255
            matrix = confusion_for(result[name], target)
            metrics[name] = {
                "image_miou": image_miou(result[name], target),
                "pixel_accuracy": (float(matrix.diag().sum() / matrix.sum())
                                   if matrix.sum() > 0 else None),
                "wrong_pixels": int(((result[name] != target) & valid).sum()),
                "confusion": matrix.tolist(),
            }

    # Seven panels; local predictions are shown as a montage.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    axes[0].imshow(image); axes[0].set_title("Original")
    if target is not None:
        axes[1].imshow(colorize(target)); axes[1].set_title("GT — gray: ignore")
    else:
        axes[1].text(0.5, 0.5, "Ground truth unavailable", ha="center", va="center")
        axes[1].set_title("Ground truth")
    axes[2].imshow(colorize(result["baseline"])); axes[2].set_title("Global prediction")
    im = axes[3].imshow(result["entropy"].numpy(), cmap="inferno", vmin=0, vmax=1)
    axes[3].set_title("Normalized entropy")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    axes[4].imshow(image); axes[4].set_title(f"Top-K crops — {len(tiles)} selected")
    for region in result["regions"]:
        x1, y1, x2, y2 = region["original_box"]
        color = plt.get_cmap("tab10")((region["rank"] - 1) % 10)
        axes[4].add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                    fill=False, edgecolor=color, linewidth=2))
        axes[4].text(x1, y1, str(region["rank"]), color=color, backgroundcolor="black")
    axes[5].imshow(montage); axes[5].set_title("Local predictions")
    axes[6].imshow(colorize(result["final"])); axes[6].set_title("Final segmentation")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(source_name)
    fig.tight_layout()
    fig.savefig(output_dir / "demo_7_panels.png", dpi=150)
    plt.show()
    plt.close(fig)

    report = {"source": source_name, "image_size": list(image.size),
              "parameters": result["parameters"],
              "regions": result["regions"], "candidates": result["candidates"],
              "passes": {"baseline": 1, "adaptive": 1 + len(tiles)},
              "union_area_percent": float((result["coverage"] > 0).float().mean() * 100),
              "metrics": metrics,
              "metric_note": "Native resolution, ignore=255; no metrics without GT.",
              "fusion": "Mean local logits inside crop union; global outside.",
              "checkpoint": str(checkpoint_path),
              "model_name": model_name, "revision": revision}
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if metrics:
        print("Baseline image mIoU:", metrics["baseline"]["image_miou"])
        print("Adaptive image mIoU:", metrics["final"]["image_miou"])
    else:
        print("GT yok; doğruluk metrikleri hesaplanmadı.")
    print("Requested / selected K:", result["parameters"]["k_requested"], "/", len(tiles))
    print("Backbone passes:", 1 + len(tiles))
    print("Saved:", output_dir)
    return report
