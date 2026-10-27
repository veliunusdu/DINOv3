"""Validate thesis benchmark/cost rows against full-precision saved evidence."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
path=ROOT/"Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md"
text=path.read_text(encoding="utf-8")
benchmark=text.split("### 4.4. Benchmark:")[1].split("### 4.5.")[0]
cost=text.split("### 4.5. Computational Cost:")[1].split("### 4.6.")[0]
quality=json.loads((ROOT/"Reproducibility/metrics_verification.json").read_text(encoding="utf-8"))["pooled_miou"]
figure_path=ROOT/"Deliverables/Quality_Cost/figure_data.json"
figure=json.loads(figure_path.read_text(encoding="utf-8"))
assert hashlib.sha256((ROOT/figure["quality_source"]).read_bytes()).hexdigest()==figure["quality_source_sha256"]
archive=ROOT/figure["timing_archive"]
assert hashlib.sha256(archive.read_bytes()).hexdigest()==figure["timing_archive_sha256"]
prefix="benchmark_all_k3_100_20261007T194714_806084Z/"
with zipfile.ZipFile(archive) as z:
    summary=json.loads(z.read(prefix+"results.json"))["summary"]
    raw=z.read(prefix+"measurements.npz")
    bootstrap=json.loads(z.read(next(n for n in z.namelist() if "paired_latency_bootstrap" in n)))
with np.load(io.BytesIO(raw),allow_pickle=False) as state:
    assert state["completed"].all()
    pilot=json.loads((ROOT/"Reproducibility/splits/latency_100.json").read_text(encoding="utf-8"))
    assert state["image_indices"].tolist()==[r["image_index"] for r in pilot["records"]]
    for i,name in enumerate(state["methods"].tolist()):
        cm=state["confusions"][i].sum(0)
        union=cm.sum(0)+cm.sum(1)-np.diag(cm)
        assert abs(float(np.mean(np.diag(cm)[union>0]/union[union>0]))-summary[name]["pooled_miou"])<1e-12
        assert abs(float(state["latency_ms"][i].mean())-summary[name]["mean_ms"])<1e-9
def f(v,n=4): return f"{v:.{n}f}".replace(".",",").replace("-","−")
names={"baseline":"Baseline","adaptive_k1":"Adaptive K=1","adaptive_k3":"Adaptive K=3",
       "random_42_k3":"Random 42 K=3","random_43_k3":"Random 43 K=3","random_44_k3":"Random 44 K=3",
       "random_k3_mean":"Random K=3 mean","sliding":"Sliding"}
for name,label in names.items():
    passes="Her seed için 4" if name=="random_k3_mean" else str(summary[name]["passes"])
    delta=(quality[name]-quality["baseline"])*100
    delta_text=("+" if delta>0 else "")+f(delta,2)
    expected=f"| {label} | {f(quality[name])} | {delta_text} | {passes} |"
    assert expected in benchmark,expected
for name,row in summary.items():
    expected=f"| {names[name]} | {f(row['pooled_miou'])} | {f(row['mean_ms'],1)} | {f(row['median_ms'],1)} | {f(row['p95_ms'],1)} |"
    assert expected in cost,expected
base=summary["baseline"]["mean_ms"]
for name in ["baseline","adaptive_k1","adaptive_k3","random_k3_mean","sliding"]:
    if name=="random_k3_mean":
        ms=sum(summary[f"random_{s}_k3"]["mean_ms"] for s in [42,43,44])/3
        passes="Her seed için 4"; extra="Her seed için 3"
    else:
        ms=summary[name]["mean_ms"]; p=summary[name]["passes"]
        passes=str(p); extra=str(p-1)
    ratio=ms/base
    expected=f"| {names[name]} | {f(ms,1)} | {passes} | {extra} | {f(ratio,2)}× | %{f((ratio-1)*100,2)} |"
    assert expected in cost,expected
assert bootstrap["bootstrap_samples"]==5000
assert "C_{rel}(m)=\\frac{\\bar t_m}{\\bar t_b}" in cost
assert "Relative cost boyutsuz ortalama süre oranıdır" in cost
assert "Şekil M" in cost and "aynı 100 görüntüden gelir" in cost
for target in re.findall(r'\]\(([^)]+)\)',text):
    if not target.startswith(("http://","https://","#")):
        assert (path.parent/target).is_file(),target
proof={"status":"PASS","date":"2026-10-27","thesis_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
       "checks":{"quality_rows_1399":8,"same_cohort_runtime_rows_100":7,"relative_cost_rows":5,
                 "raw_100_confusions_and_latencies_recomputed":True,"latency_ids_match":True,"cohort_separation_present":True,
                 "relative_cost_is_mean_latency_ratio":True,"figure_and_source_hashes":True,"local_links":True},
       "sources":{"quality":"Reproducibility/metrics_verification.json","timing_archive":figure["timing_archive"],
                  "timing_archive_sha256":figure["timing_archive_sha256"],"figure_data_sha256":hashlib.sha256(figure_path.read_bytes()).hexdigest()},
       "scope":"No inference, retraining, timing or bootstrap rerun. Raw measurement aggregates checked; existing bootstrap intervals retained."}
(ROOT/"Reproducibility/thesis_benchmark_cost_audit_2026-10-27.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("PASS: 8 main-quality rows, 7 same-100 runtime rows, 5 full-precision cost ratios, cohort scope and figure sources.")
