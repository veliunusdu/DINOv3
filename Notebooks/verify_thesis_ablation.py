"""Verify newly consolidated thesis tables against saved ablation JSON records."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parent.parent
path=ROOT/"Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md"
text=path.read_text(encoding="utf-8")
section=text.split("### 4.7. Region Selector Ablation")[1].split("### 4.8.")[0]
archive=ROOT/"Downloads/DINOv3_complete_20261008T122013_722899Z.zip"
prefix="20261001T181030_523202Z/"
sources=[]
def f(value,digits=4): return f"{value:.{digits}f}".replace(".",",")
with zipfile.ZipFile(archive) as z:
    def load(member):
        raw=z.read(prefix+member)
        sources.append({"member":prefix+member,"sha256":hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    k=load("topk_comparison_50_20261006T223417_739370Z/results.json")
    for count in [1,2,3]:
        row=k["comparisons"][f"k{count}"]
        adaptive=k["summary"][f"adaptive_k{count}"]
        wins="/".join(map(str,adaptive["improved_worsened_unchanged"]))
        expected=f"| {count} | {f(adaptive['pooled_miou'])} | {f(row['random_mean_miou'])} ± {f(row['random_seed_std_miou'])} | +{f(row['adaptive_minus_random_points'],2)} | {wins} | {count+1} |"
        assert expected in section,expected
    sizes=load("crop_size_comparison_50_20261006T210506_222816Z.json")
    for size in [56,84,112]:
        row=sizes["comparisons"][str(size)]
        expected=f"| {size} | {f(row['adaptive_miou'])} | {f(row['random_mean_miou'])} ± {f(row['random_seed_std'])} | +{f(row['adaptive_minus_random']*100,2)} |"
        assert expected in section,expected
    threshold=load("selector_threshold_ablation_dev50_v1/results.json")
    labels={"reference":"Reference: entropy=0, IoU=0,30","entropy_0.10":"Minimum entropy=0,10",
            "entropy_0.20":"Minimum entropy=0,20","entropy_0.30":"Minimum entropy=0,30",
            "overlap_0.10":"IoU eşiği=0,10","overlap_0.50":"IoU eşiği=0,50"}
    for name,label in labels.items():
        row=threshold["summary"][name]
        expected=f"| {label} | {f(row['pooled_miou'])} | {f(row['mean_selected_k'],2)} | {f(row['mean_standalone_passes'],2)} |"
        assert expected in section,expected
    protocol=load("selector_threshold_ablation_dev50_v1/protocol.json")
    assert protocol["k_max"]==3 and protocol["crop_size"]==112 and protocol["stride"]==28
    assert sizes["protocol"]["K"]==1
    assert "prefix" in k["protocol"]["random_selection"]
for phrase in ["0,7104","0,7133","0,4883","0,6180","%59,88","224×224","1.449","1.399"]:
    assert phrase in text
for name in ["Baseline segmentasyon sonuçları","Oracle Zoom: hedef kutusu bilindiğinde yakınlaştırma",
             "Uncertainty estimation: entropy ve hata ilişkisi","Region Selector Ablation"]:
    assert name in text
for target in re.findall(r'\]\(([^)]+)\)',text):
    if not target.startswith(("https://","http://","#")):
        assert (path.parent/target).is_file(),target
result={"status":"PASS","scope":"Top-K, crop-size and threshold tables matched to saved JSON; section/link checks.",
        "thesis_path":path.relative_to(ROOT).as_posix(),"thesis_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_archive":archive.relative_to(ROOT).as_posix(),"source_members":sources,
        "reported_only_results":{"baseline_1449_224":0.7104,"oracle_development_50":0.6180,"entropy_top20_error_recall_percent":59.88},
        "limits":"Early baseline, oracle and pixel entropy numbers are historical reported outputs, not newly recomputed here. No new inference or statistical intervals."}
(ROOT/"Reproducibility/thesis_ablation_audit_2026-10-20.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("PASS: 3 Top-K rows, 3 crop-size rows, 6 threshold rows; sections and local links verified.")
