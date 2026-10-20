"""Plot saved results, recomputing the 100-image measurements before rendering."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
local_dependencies = ROOT / "Artifacts/plot_dependencies"
if local_dependencies.exists():
    sys.path.insert(0, str(local_dependencies))
import numpy as np
cache = ROOT / "tmp/matplotlib_cache"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "Deliverables/Quality_Cost"
OUT.mkdir(parents=True, exist_ok=True)
archive = ROOT / "Downloads/benchmark_all_k3_100_20261007T194714_806084Z (1).zip"
prefix = "benchmark_all_k3_100_20261007T194714_806084Z/"
with zipfile.ZipFile(archive) as z:
    raw = z.read(prefix + "measurements.npz")
    summary = json.loads(z.read(prefix + "results.json"))["summary"]
    protocol = json.loads(z.read(prefix + "protocol.json"))
with np.load(io.BytesIO(raw), allow_pickle=False) as state:
    assert state["completed"].all()
    assert state["image_indices"].tolist() == protocol["image_indices"]
    assert len(set(protocol["image_indices"])) == 100
    methods = state["methods"].tolist()
    results = {}
    for i, method in enumerate(methods):
        cm = state["confusions"][i].sum(axis=0)
        union = cm.sum(0)+cm.sum(1)-np.diag(cm)
        miou = float(np.mean(np.diag(cm)[union>0]/union[union>0]))
        ms = float(state["latency_ms"][i].mean())
        assert abs(miou-summary[method]["pooled_miou"]) < 1e-12
        assert abs(ms-summary[method]["mean_ms"]) < 1e-9
        results[method] = {"pooled_miou":miou,"mean_ms":ms,"passes":summary[method]["passes"]}
results["random_k3_mean"] = {
    "pooled_miou":float(np.mean([results[f"random_{s}_k3"]["pooled_miou"] for s in [42,43,44]])),
    "mean_ms":float(np.mean([results[f"random_{s}_k3"]["mean_ms"] for s in [42,43,44]])),
    "passes":4,
}
verified = json.loads((ROOT / "Reproducibility/metrics_verification.json").read_text(encoding="utf-8"))
assert verified["status"] == "PASS"
full = verified["pooled_miou"]
ordered = ["baseline","adaptive_k1","adaptive_k3","random_42_k3","random_43_k3","random_44_k3","sliding"]
labels = {"baseline":"Baseline", "adaptive_k1":"Adaptive K=1", "adaptive_k3":"Adaptive K=3",
          "random_42_k3":"Random K=3 (42)","random_43_k3":"Random K=3 (43)",
          "random_44_k3":"Random K=3 (44)","sliding":"Sliding-window","random_k3_mean":"Random K=3 mean"}
colors = {"baseline":"#203747","adaptive_k1":"#2672AA","adaptive_k3":"#008878", "sliding":"#A44D19"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,6.8),gridspec_kw={"width_ratios":[1.4,1]})
for method in ordered:
    value=results[method]
    ax.scatter(value["mean_ms"],value["pooled_miou"],s=80 if "random" not in method else 45,
               color=colors.get(method,"#888888"),marker="o" if "random" not in method else "x",zorder=3)
offsets={"baseline":(8,-18),"adaptive_k1":(-24,-25),"adaptive_k3":(12,-45),
         "sliding":(-104,-18),"random_42_k3":(12,0),"random_43_k3":(12,3),"random_44_k3":(12,-8)}
for method in ordered:
    v=results[method]
    ax.annotate(labels[method],(v["mean_ms"],v["pooled_miou"]),xytext=offsets[method],textcoords="offset points",
                color=colors.get(method,"#666666"),fontsize=10,
                arrowprops={"arrowstyle":"-","color":colors.get(method,"#666666"),"lw":.7}
                if method in ["adaptive_k1","adaptive_k3"] else None)
ax.set(xlim=(50,565),ylim=(.644,.687),xlabel="Mean latency (ms/image), Tesla T4",ylabel="Pooled mIoU (0–1)",
       title="A. Same 100 images: quality and cost")
ax.grid(alpha=.2)
ax.text(.98,.03,"Higher quality ↑   Lower latency ←",transform=ax.transAxes,ha="right",fontsize=10,color="#555555")
quality_order=["baseline","adaptive_k1","adaptive_k3","random_k3_mean","sliding"]
for i,method in enumerate(quality_order):
    value=full[method]
    bx.scatter(value,i,s=90,color=colors.get(method,"#888888"),zorder=3)
    bx.annotate(f"{value:.4f}",(value,i),xytext=(9,0),textcoords="offset points",va="center",fontsize=11)
bx.set_yticks(range(len(quality_order)),[labels[m] for m in quality_order])
bx.set(xlim=(.705,.744),ylim=(4.5,-.5),xlabel="Pooled mIoU (0–1)",title="B. Main evaluation: 1,399 images")
bx.grid(axis="x",alpha=.2)
bx.text(.02,.02,"Quality only; no 1,399-image latency measured",transform=bx.transAxes,fontsize=9,color="#555555")
fig.suptitle("DINOv3 Adaptive Zoom: quality–cost evidence",fontsize=18,fontweight="bold",y=.97)
fig.text(.06,.075,"Panel A: raw same-session measurements, serial sliding, 3 fixed random seeds. Random points are separate runs.",fontsize=10)
fig.text(.06,.045,"Panel B: native-resolution validation excluding dev50; contains the 100-image pilot. Random mean is not an ensemble.",fontsize=10)
fig.text(.06,.015,"Axes are zoomed to show differences. No error bars: this figure shows point estimates, not significance tests.",fontsize=10)
fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.22,wspace=.49)
for ext in ["png","svg"]:
    fig.savefig(OUT / f"quality_cost.{ext}",dpi=220)
svg_path=OUT/"quality_cost.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
plt.close(fig)
data={"timing_archive":archive.relative_to(ROOT).as_posix(),"timing_archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
      "measurements_member":prefix+"measurements.npz","measurements_sha256":hashlib.sha256(raw).hexdigest(),
      "timing_100":results,"quality_1399":{m:full[m] for m in quality_order},
      "quality_source":"Reproducibility/metrics_verification.json",
      "quality_source_sha256":hashlib.sha256((ROOT/"Reproducibility/metrics_verification.json").read_bytes()).hexdigest(),
      "scope":"Panel A same 100-image mIoU and latency; panel B quality only for 1399.",
      "inference_performed":False}
(OUT/"figure_data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Recomputed all seven 100-image measurements; summary match passed. Figure saved:",OUT)
