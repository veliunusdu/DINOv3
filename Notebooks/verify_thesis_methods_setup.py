"""Check the thesis methods/setup description against config, IDs, code and saved evidence."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

ROOT=Path(__file__).resolve().parent.parent
thesis=ROOT/"Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md"
text=thesis.read_text(encoding="utf-8")
method=text.split("## 2. Yöntem")[1].split("## 4. Sonuçlar")[0]
config=tomllib.loads((ROOT/"Config/DINOv3_Final_Config.toml").read_text(encoding="utf-8-sig"))
for name,value in [("k",3),("crop_size",112),("stride",28),("minimum_entropy",0),("overlap_iou",.3),("padding",0)]:
    assert config["selector"][name]==value
    assert f"`{name}`" in method
assert config["checkpoint"]["sha256"] in method
assert config["model"]["revision"] in text
assert "mean_local_logits_global_outside"==config["fusion"]["mode"]
assert "align_corners=False" in method
splits={}
for name,count in [("validation_1449",1449),("development_50",50),("evaluation_1399",1399),("latency_100",100)]:
    path=ROOT/f"Reproducibility/splits/{name}_ids.txt"
    ids=path.read_text(encoding="utf-8").splitlines()
    assert len(ids)==len(set(ids))==count
    splits[name]=set(ids)
assert not splits["development_50"] & splits["evaluation_1399"]
assert splits["development_50"] | splits["evaluation_1399"]==splits["validation_1449"]
assert splits["latency_100"] <= splits["evaluation_1399"]
tree=ast.parse((ROOT/"Notebooks/adaptive_zoom_demo.py").read_text(encoding="utf-8"))
pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ["box_iou","topk_box_to_original"]],type_ignores=[])
ns={"math":math}
exec(compile(pure,"documented-coordinate-check","exec"),ns)
assert ns["topk_box_to_original"]([0,112,112,224],(500,375))==[0,187,250,375]
assert ns["topk_box_to_original"]([0,0,224,224],(333,501))==[0,0,333,501]
assert ns["box_iou"]([0,0,10,10],[10,0,20,10])==0
clean=json.loads((ROOT/"Reproducibility/clean_colab_verification_2026-10-20.json").read_text(encoding="utf-8"))
assert clean["status"]=="PASS"
env=clean["environment"]
for value in [env["python"],env["torch"],env["packages"]["torchvision"],env["packages"]["transformers"],env["packages"]["huggingface_hub"],env["gpu"]]:
    assert value in method,value
training=json.loads((ROOT/"Reproducibility/historical_model_config.json").read_text(encoding="utf-8"))
assert (training["epochs"],training["batch_size"],training["learning_rate"],training["weight_decay"],training["seed"])==(5,8,.001,.0001,42)
for phrase in ["tam ID listesi", "orijinal Git commit", "eğitimin yeniden üretimi", "1 bağımlılıklar", "1b final config", "GT'ye başvurulmaz"]:
    assert phrase in method,phrase
for target in re.findall(r'\]\(([^)]+)\)',text):
    if not target.startswith(("https://","http://","#")):
        assert (thesis.parent/target).is_file(),target
sources=["Config/DINOv3_Final_Config.toml","Notebooks/adaptive_zoom_demo.py","Reproducibility/historical_model_config.json",
         "Reproducibility/clean_colab_verification_2026-10-20.json","Reproducibility/experiment_manifest.json"]
proof={"status":"PASS","date":"2026-10-27","scope":"Document/config values, saved split relations, pure coordinate functions, training metadata and clean-run environment; no new GPU inference.",
       "thesis_sha256":hashlib.sha256(thesis.read_bytes()).hexdigest(),
       "sources":[{"path":p,"sha256":hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources],
       "checks":{"final_config":True,"split_counts_and_relations":True,"coordinate_examples":True,"environment":True,
                 "recorded_training_metadata":True,"limitations_present":True,"local_links":True}}
(ROOT/"Reproducibility/thesis_methods_setup_audit_2026-10-27.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("PASS: methods/setup config, split counts/relations, coordinate examples, environment, metadata and links.")
