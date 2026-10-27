"""Recompute archived pooled mIoU. No model, GPU, or inference required."""
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Reproducibility"
ARCHIVE = "Downloads/DINOv3_complete_20261008T122013_722899Z.zip"
PREFIX = "20261001T181030_523202Z/"
FULL_TOL = 1e-12
DISPLAY_TOL = 0.000050000001  # Four-decimal displayed mIoU rounding.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    import zipfile
    manifest = json.loads((OUT / "experiment_manifest.json").read_text(encoding="utf-8"))
    expected_archive = next(a for a in manifest["source_archives"] if a["path"] == ARCHIVE)
    assert sha((ROOT / ARCHIVE).read_bytes()) == expected_archive["sha256"]
    sources = []
    scores = {}
    matrices = {}
    totals = None
    indices = None
    with zipfile.ZipFile(ROOT / ARCHIVE) as z:
        def read(member):
            data = z.read(PREFIX + member)
            sources.append({"archive": ARCHIVE, "member": PREFIX + member, "sha256": sha(data)})
            return data
        fixed_protocol = json.loads(read("full_voc_fixed_v1/protocol.json"))
        k3_protocol = json.loads(read("full_voc_topk3_v1/protocol.json"))
        fixed_summary = json.loads(read("full_voc_fixed_v1/summary.json"))
        k3_summary = json.loads(read("full_voc_topk3_v1/summary.json"))
        assert fixed_protocol["evaluation_resolution"] == "original"
        assert k3_protocol["evaluation_resolution"] == "native"
        assert fixed_protocol["ignore_index"] == k3_protocol["ignore_label"] == 255
        assert fixed_protocol["background_included"] is True
        assert fixed_protocol["checkpoint_sha256"] == k3_protocol["checkpoint_sha256"] == manifest["checkpoint"]["sha256"]
        assert fixed_summary["status"] == k3_summary["status"] == "complete"
        for folder, protocol in [("full_voc_fixed_v1",fixed_protocol),("full_voc_topk3_v1",k3_protocol)]:
            with np.load(io.BytesIO(read(folder + "/per_image_state.npz")), allow_pickle=False) as state:
                methods = state["methods"].tolist()
                cm = state["confusions"]
                assert methods == protocol["methods"]
                assert cm.shape == (len(methods),1399,21,21)
                assert np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
                assert state["completed"].shape == (1399,) and state["completed"].all()
                if indices is None:
                    indices = state["image_indices"].copy()
                else:
                    assert np.array_equal(indices,state["image_indices"])
                assert indices.tolist() == k3_protocol["image_indices"]
                for i, name in enumerate(methods):
                    canonical = "adaptive_k1" if name == "adaptive" else name
                    count = cm[i].sum(axis=(1,2),dtype=np.int64)
                    assert (count > 0).all()
                    if totals is None:
                        totals = count.copy()
                    else:
                        assert np.array_equal(totals,count), "Methods do not evaluate identical valid-pixel counts."
                    pooled = cm[i].sum(axis=0,dtype=np.int64)
                    tp = np.diag(pooled).astype(np.float64)
                    union = pooled.sum(0)+pooled.sum(1)-np.diag(pooled)
                    valid = union > 0
                    assert valid.all(), "All 21 classes expected in full pooled evaluation."
                    scores[canonical] = float((tp[valid]/union[valid]).mean())
                    matrices[canonical] = pooled.tolist()
    eval_ids = json.loads((OUT / "splits/evaluation_1399.json").read_text(encoding="utf-8"))["records"]
    assert [r["image_index"] for r in eval_ids] == indices.tolist()
    assert [r["image_id"] for r in eval_ids] == fixed_protocol["image_ids"]
    comparisons = []
    for name, value in fixed_summary["pooled_miou"].items():
        canonical = "adaptive_k1" if name == "adaptive" else name
        delta = scores[canonical]-value
        assert abs(delta) <= FULL_TOL
        comparisons.append({"method":canonical,"source":"fixed_summary","recorded":value,"recomputed":scores[canonical],"difference":delta})
    for name, value in k3_summary["summary"].items():
        delta = scores[name]-value
        assert abs(delta) <= FULL_TOL
        comparisons.append({"method":name,"source":"k3_summary","recorded":value,"recomputed":scores[name],"difference":delta})
    random_mean = float(np.mean([scores[f"random_{seed}_k3"] for seed in [42,43,44]]))
    assert abs(random_mean-k3_summary["random_k3_mean_miou"]) <= FULL_TOL
    scores["random_k3_mean"] = random_mean
    csv_mapping = {"Baseline":"baseline","Sliding-window":"sliding","Adaptive K=1":"adaptive_k1",
                   "Adaptive K=3":"adaptive_k3","Random K=3 three-seed mean":"random_k3_mean",
                   **{f"Random {s} K=3":f"random_{s}_k3" for s in [42,43,44]}}
    with (ROOT / "Deliverables/DINOv3_Method_Comparison.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        name = csv_mapping[row["Method"]]
        assert abs(scores[name]-float(row["mIoU"])) <= DISPLAY_TOL
        assert abs((scores[name]-scores["baseline"])*100-float(row["Delta mIoU"])) <= 0.0050000001
    report = (ROOT / "Docs/DINOv3_Final_Deney_Raporu_TR.md").read_text(encoding="utf-8")
    section = report.split("## Ana doğruluk değerlendirmesi — 1.399 görüntü")[1].split("## Son süre benchmark")[0]
    labels = {"Baseline":"baseline","Adaptive K=1":"adaptive_k1","Adaptive K=3":"adaptive_k3",
              "Sliding":"sliding","Random K=3 üç seed mean":"random_k3_mean",
              **{f"Random {s} K=3":f"random_{s}_k3" for s in [42,43,44]}}
    checked = set()
    for line in section.splitlines():
        if line.startswith("|"):
            cells = [x.strip() for x in line.split("|")[1:-1]]
            if cells[0] in labels:
                name = labels[cells[0]]
                assert abs(scores[name]-float(cells[1].replace(",","."))) <= DISPLAY_TOL
                checked.add(name)
    assert checked == set(labels.values())
    result = {"status":"PASS","created_at":datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
              "images":1399,"classes":21,"metric":"sum image confusion matrices, class IoU mean, background included",
              "full_precision_tolerance":FULL_TOL,"display_rounding_tolerance":DISPLAY_TOL,
              "pooled_miou":scores,"summary_comparisons":comparisons,"source_members":sources,
              "archive_sha256":expected_archive["sha256"],"script_sha256":sha(Path(__file__).read_bytes()),
              "checks":{"same_ordered_image_indices":True,"same_image_ids":True,"all_completed":True,
                        "same_valid_pixel_counts":True,"archive_summary_exact_match":True,
                        "report_rounded_match":True,"csv_rounded_match":True},
              "limits":["No GPU inference or bootstrap rerun.","Native resolution and ignore=255 verified as protocol metadata; raw masks not reloaded.",
                        "0.7104 belongs to a separate 1449-image 224x224 evaluation, not this native-resolution comparison."]}
    (OUT / "metrics_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# Kayıtlı mIoU doğrulama raporu", "", "**Durum: PASS.** Arşivdeki confusion matrix'lerden pooled mIoU yeniden hesaplandı; yeni inference yapılmadı.",
             "", "Kapsam: aynı sıradaki 1.399 görüntü, 21 sınıf ve background dahil. Native çözünürlük ve ignore=255 protokol kaydından doğrulandı; ham GT maskeleri tekrar işlenmedi.",
             "", "| Yöntem | Yeniden hesaplanan mIoU | Baseline farkı (puan) |", "|---|---:|---:|"]
    for name in ["baseline","adaptive_k1","adaptive_k3","random_42_k3","random_43_k3","random_44_k3","random_k3_mean","sliding"]:
        lines.append(f"| {name} | {scores[name]:.10f} | {(scores[name]-scores['baseline'])*100:+.4f} |")
    lines += ["", "Arşiv summary dosyalarıyla tam hassasiyet karşılaştırması 1e-12 toleransında geçti. Final rapor ve CSV'nin dört ondalıklı mIoU değerleri yuvarlama toleransında eşleşti. CSV delta değerleri tam hassasiyetli sonuçlardan hesaplanıp iki ondalık yüzde puanla gösterilir.",
              "", "İki state dosyasının görüntü indeksleri, tamamlanma işaretleri, method listeleri ve yöntem başına geçerli piksel sayıları kontrol edildi. Random mean üç seed'in ayrı mIoU değerlerinin ortalamasıdır; ensemble değildir.",
              "", "**0,7104 ile karşılaştırma yapılmadı:** bu değer ayrı 1.449 görüntü/224×224 değerlendirmesine aittir. Bootstrap güven aralıkları ve süre ölçümleri bu görevde yeniden hesaplanmadı.",
              "", "Kaynak üyeleri ve hash'ler `metrics_verification.json` içinde. Tekrar çalıştırmak için proje kökünde `python Notebooks/verify_saved_metrics.py` kullan; NumPy ve yerel kaynak arşiv gerekir."]
    (OUT / "metrics_verification_TR.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("PASS: archived summaries, final report and CSV agree with recomputed pooled mIoU.")
    for name in ["baseline","adaptive_k1","adaptive_k3","random_k3_mean","sliding"]:
        print(name, f"{scores[name]:.10f}")


if __name__ == "__main__":
    verify()
