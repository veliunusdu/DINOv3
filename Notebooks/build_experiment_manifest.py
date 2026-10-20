"""Build provenance from saved archives, without loading torch or running inference."""
import hashlib
import io
import json
import platform
import subprocess
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Reproducibility"
OUT.mkdir(exist_ok=True)
PREFIX = "20261001T181030_523202Z/"
MAIN = "Downloads/DINOv3_complete_20261008T122013_722899Z.zip"
TIME = "Downloads/benchmark_all_k3_100_20261007T194714_806084Z (1).zip"
TP = "benchmark_all_k3_100_20261007T194714_806084Z/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_info(path):
    p = ROOT / path
    return {"path": path, "size_bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}


members = []


def member_bytes(archive, member):
    with zipfile.ZipFile(ROOT / archive) as z:
        data = z.read(member)
    info = {"archive": archive, "member": member, "sha256": digest(data), "size_bytes": len(data)}
    if info not in members:
        members.append(info)
    return data


def saved_json(archive, member):
    return json.loads(member_bytes(archive, member))


def write_json(path, data):
    (OUT / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


config = tomllib.loads((ROOT / "Config/DINOv3_Final_Config.toml").read_text(encoding="utf-8-sig"))
checkpoint = member_bytes(MAIN, config["checkpoint"]["member"])
assert digest(checkpoint) == config["checkpoint"]["sha256"]
historical_config = saved_json(MAIN, PREFIX + "config.json")
assert historical_config["model_name"] == config["model"]["id"]
assert historical_config["revision"] == config["model"]["revision"]
fixed = saved_json(MAIN, PREFIX + "full_voc_fixed_v1/protocol.json")
k3 = saved_json(MAIN, PREFIX + "full_voc_topk3_v1/protocol.json")
dev = saved_json(MAIN, PREFIX + "adaptive_random_50_20261006T204801_022252Z.json")
holdout = saved_json(MAIN, PREFIX + "holdout_split_100_20261006T212410_602431Z.json")
timing = saved_json(TIME, TP + "protocol.json")
threshold = saved_json(MAIN, PREFIX + "selector_threshold_ablation_dev50_v1/protocol.json")
main_summary = saved_json(MAIN, PREFIX + "full_voc_fixed_v1/summary.json")
k3_summary = saved_json(MAIN, PREFIX + "full_voc_topk3_v1/summary.json")
timing_summary = saved_json(TIME, TP + "results.json")
assert fixed["checkpoint_sha256"] == k3["checkpoint_sha256"] == timing["checkpoint_sha256"] == digest(checkpoint)

# Recover index -> image ID from explicit archived records, not guessed filenames.
indices = k3["image_indices"]
assert len(indices) == len(fixed["image_ids"]) == 1399
mapping = dict(zip(indices, fixed["image_ids"]))
dev_records = [{"image_index": r["image_index"], "image_id": r["image_id"]} for r in dev["records"]]
assert len(dev_records) == 50
dev_indices = [r["image_index"] for r in dev_records]
assert dev_indices == fixed["excluded_development_indices"] == threshold["image_indices"] == holdout["development_indices"]
assert not set(indices) & set(dev_indices)
mapping.update({r["image_index"]: r["image_id"] for r in dev_records})
assert set(mapping) == set(range(1449)) and len(set(mapping.values())) == 1449
assert [mapping[i] for i in range(1449)] == sorted(mapping.values())
assert timing["image_indices"] == holdout["holdout_indices"]
assert [mapping[i] for i in holdout["holdout_indices"]] == holdout["image_ids"]
assert set(timing["image_indices"]).issubset(indices)
for folder, expected_methods in [("full_voc_fixed_v1", fixed["methods"]), ("full_voc_topk3_v1", k3["methods"])]:
    payload = member_bytes(MAIN, PREFIX + folder + "/per_image_state.npz")
    with np.load(io.BytesIO(payload), allow_pickle=False) as state:
        assert state["image_indices"].tolist() == indices
        assert state["methods"].tolist() == expected_methods
        assert state["completed"].all()
        assert state["confusions"].shape == (len(expected_methods),1399,21,21)

split_specs = {
    "validation_1449": (list(range(1449)), "Reconstructed union of explicitly recorded evaluation and development index/ID pairs."),
    "development_50": (dev_indices, "Explicit adaptive_random_50 records; excluded from main evaluation."),
    "evaluation_1399": (indices, "Indices from full_voc_topk3_v1 and image IDs from full_voc_fixed_v1; ordered NPZ indices cross-checked."),
    "latency_100": (timing["image_indices"], "Explicit latest 7-method timing protocol; same IDs as saved holdout split."),
}
splits = {}
for name, (selection, origin) in split_specs.items():
    records = [{"image_index": i, "image_id": mapping[i]} for i in selection]
    json_name = f"splits/{name}.json"
    text_name = f"splits/{name}_ids.txt"
    (OUT / "splits").mkdir(exist_ok=True)
    write_json(json_name, {"dataset": "VOC2012 validation", "index_base": 0, "origin": origin, "records": records})
    (OUT / text_name).write_text("\n".join(mapping[i] for i in selection) + "\n", encoding="utf-8")
    splits[name] = {"count": len(selection), "origin": origin,
                   "records": file_info("Reproducibility/" + json_name),
                   "ids": file_info("Reproducibility/" + text_name)}
splits["training_1464"] = {"count_reported": 1464, "ids": None,
                          "status": "Exact training ID list not found in inspected archives; not reconstructed from assumptions."}

# Distinguish a post-experiment Git commit from the uncommitted current demo.
git_head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
code_paths = ["Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb", "Notebooks/adaptive_zoom_demo.py",
              "Notebooks/build_final_demo_notebook.py", "Notebooks/final_config.py",
              "Notebooks/verify_final_config.py", "Notebooks/build_experiment_manifest.py",
              "Config/DINOv3_Final_Config.toml", "README.md", "Notebooks/Final_Demo_Kullanim_TR.md"]
code_files = []
for path in code_paths:
    info = file_info(path)
    snapshot = OUT / "code_snapshot" / path
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((ROOT / path).read_bytes())
    info["snapshot_path"] = snapshot.relative_to(ROOT).as_posix()
    code_files.append(info)

archives = []
for path in sorted((ROOT / "Downloads").glob("*.zip")):
    info = file_info(path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:
        info["member_count"] = len(z.namelist())
        info["historical_environment_members"] = [n for n in z.namelist() if "environment" in n.lower() and n.endswith(".json")]
    archives.append(info)

latency_bootstrap_member = next(n for n in zipfile.ZipFile(ROOT / TIME).namelist() if "paired_latency_bootstrap" in n)
latency_bootstrap = saved_json(TIME, latency_bootstrap_member)
bootstrap_path = "Downloads/paired_bootstrap_k3_1399_20261006T230527_678459Z.json"
bootstrap = json.loads((ROOT / bootstrap_path).read_text(encoding="utf-8"))
source_files = [file_info(p) for p in [bootstrap_path, "Downloads/DINOv3.ipynb",
              "Docs/DINOv3_Final_Deney_Raporu_TR.md", "Deliverables/DINOv3_Method_Comparison.csv"]]
code_sources = {"git_head_at_manifest_creation": git_head,
                "historical_experiment_git_commit": None,
                "historical_commit_note": "Experiments preceded Git initialization; current HEAD is not their original execution commit.",
                "current_code_status": "Config-connected demo has uncommitted changes; exact files preserved by SHA-256 and code_snapshot.",
                "files": code_files,
                "historical_notebook": "Downloads/DINOv3.ipynb; archived code, not certified as a clean sequential reproduction of all original runs."}
manifest = {
    "schema_version": 1, "created_at": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
    "scope": "Saved experiment provenance plus current demo snapshot; no GPU inference or retraining performed.",
    "model": config["model"], "checkpoint": {**config["checkpoint"], "verified_against_archive": True},
    "config": file_info("Config/DINOv3_Final_Config.toml"), "code": code_sources,
    "dataset": {"name": "VOC2012", "splits": splits,
                "scope_note": "Development is GT-selected. Latency pilot is part of previously seen evaluation_1399, not an untouched test."},
    "seeds": {"base": {"value": historical_config["seed"], "source": PREFIX+"config.json"},
              "random_k3": {"values": k3["random_seeds"], "selection_rule": k3["random_selection"]},
              "pilot_selection": {"value": holdout["selection_seed"], "source": PREFIX+"holdout_split_100_20261006T212410_602431Z.json"},
              "quality_bootstrap": {"seed": bootstrap["bootstrap_seed"], "bootstrap_samples": bootstrap["bootstrap_samples"], "source": bootstrap_path},
              "latency_bootstrap": {k: latency_bootstrap[k] for k in ("seed","bootstrap_samples") if k in latency_bootstrap}},
    "environments": {"historical_latency_recorded": {k: timing.get(k) for k in ("gpu","torch","transformers")},
                     "historical_missing": ["Python version", "CUDA runtime version", "torchvision version", "complete pip freeze", "original training environment"],
                     "source": {"archive": TIME, "member": TP+"protocol.json"},
                     "manifest_builder_only": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
                     "note": "Local builder versions are not the Colab experiment environment. No new clean GPU run is claimed."},
    "results": {"main_quality": {"archive": MAIN, "members": [PREFIX+"full_voc_fixed_v1/summary.json",PREFIX+"full_voc_topk3_v1/summary.json"],
                                "baseline_k1_sliding": main_summary, "k3": k3_summary},
                "latency": {"archive": TIME, "member": TP+"results.json", "summary": timing_summary["summary"]},
                "threshold_ablation": {"archive": MAIN, "member": PREFIX+"selector_threshold_ablation_dev50_v1/protocol.json"},
                "quality_bootstrap_file": bootstrap_path,
                "latency_bootstrap": {"archive": TIME, "member": latency_bootstrap_member},
                "classwise_archive": "Downloads/classwise_iou_1399_20261007T203351_459375Z.zip",
                "benchmark_csv_note": "CSV combines 1399-image quality and 100-image latency; values exported from rounded report, not raw full precision."},
    "source_archives": archives, "source_members": members, "source_files": source_files,
    "checks": {"checkpoint_hash": True, "model_revision": True, "evaluation_methods_same_ids": True,
               "development_disjoint": True, "validation_union_1449": True, "latency_subset": True,
               "npz_completed_1399": True},
    "limitations": ["Original experiment Git commit not recorded.", "Exact training ID list absent.",
                    "Historical environment record is partial.", "ZIP sources are ignored by Git and need separate backup.",
                    "Current config-connected notebook has not yet passed a fresh Colab GPU run."],
}
write_json("experiment_manifest.json", manifest)
write_json("historical_model_config.json", historical_config)
(OUT / "README_TR.md").write_text('''# Deney manifest’i

`experiment_manifest.json`, kayıtlı deneylerin checkpoint, veri, sonuç ve kod kaynaklarını birleştirir. Yeni inference yapılmadı.

## İçerik

- Checkpoint SHA-256 ve model revision arşivle doğrulandı.
- `splits/`: validation 1.449, development 50, ana değerlendirme 1.399 ve latency 100 için sıralı görüntü ID listeleri ve indeks eşleşmeleri.
- `code_snapshot/`: manifest oluşturulurken kullanılan güncel notebook, config ve yardımcı kodların birebir kopyaları. Manifest’te SHA-256 değerleri bulunur.
- `historical_model_config.json`: arşivdeki model/eğitim parametrelerinin kopyası.
- ZIP kaynaklarının dosya boyutları ve SHA-256 değerleri; okunan arşiv üyelerinin ayrı hash’leri.

## Önemli sınırlar

Güncel Git HEAD, geçmiş deneylerin çalıştırıldığı commit değildir. Config bağlantılı güncel kod henüz commit edilmemiştir; bu nedenle exact dosya kopyaları ve hash’leri saklandı. Eski notebook’un hash’i de kayıtlıdır ancak bütün deneylerin eksiksiz orijinal kod sürümü olduğu iddia edilmez.

Training 1.464 görüntünün tam ID listesi incelenen arşivlerde bulunmadı. Bu alan `null` olarak işaretlendi; tahminle doldurulmadı. Tarihsel ortamdan Tesla T4, Torch ve Transformers sürümleri timing protokolünden doğrulandı; Python/CUDA/torchvision ve tam pip freeze bu kaynakta mevcut değil. Yerel manifest üreticisinin sürümleri Colab ortamı olarak gösterilmez.

Latency 100, evaluation 1.399 içinde yer alır. Development 50 dışarıdadır. Bu gruplar dokunulmamış test olarak sunulmaz. Manifest ve hash’ler kaynakları tanımlar; tek başına yeni inference veya eğitim reproduksiyonunun kanıtı değildir.

## Yeniden üretme

Proje kökünden, NumPy içeren Python 3.11+ ile `python Notebooks/build_experiment_manifest.py` çalıştır. `Downloads/` arşivleri ve manifest’teki kaynak dosyalar yerelde bulunmalıdır. Kaynaklar veya kod değiştiğinde manifest’i yenilemeden önce önceki manifest paketini ayrı sakla. ZIP’ler GitHub’da değildir; ayrı yedek gerekir.
''', encoding="utf-8")
print("Manifest saved:", OUT / "experiment_manifest.json")
print("Verified split counts:", {k:v.get("count") for k,v in splits.items() if "count" in v})
print("Archives fingerprinted:", len(archives))
print("Historical gaps explicitly marked: training IDs, original Git commit, full environment.")
