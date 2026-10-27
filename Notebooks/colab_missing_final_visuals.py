"""Run in the existing verified Colab session; only four missing examples."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from google.colab import files

required = ["raw_val_dataset","restored_model","processor","device","CONFIG_PATH",
            "load_final_config","run_adaptive_zoom","save_and_show_demo","checkpoint_path"]
missing=[name for name in required if name not in globals()]
assert not missing, f"Önce temiz-test notebook'unda 1–5 hücrelerini çalıştır: {missing}"
active=load_final_config(CONFIG_PATH)
s=active["selector"]
assert [s[k] for k in ["k","crop_size","stride","minimum_entropy","overlap_iou","padding"]]==[3,112,28,0,.3,0]
assert hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()==active["checkpoint"]["sha256"]
assert model_name==active["model"]["id"] and revision==active["model"]["revision"]
references={13:(.3065,.5393),18:(.3875,.4703),23:(.7100,.8500),14:(.4552,.3773)}
stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
output=Path("/content")/f"DINOv3_final_visuals_missing_{stamp}"
output.mkdir()
(output/"DINOv3_Final_Config.toml").write_bytes(CONFIG_PATH.read_bytes())
report={"scope":"Four selected development examples, not a new evaluation sample.","status":"running","records":[],
        "checkpoint_sha256":active["checkpoint"]["sha256"],"config_sha256":hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()}
try:
    for index,expected in references.items():
        image,mask=raw_val_dataset[index]
        image=image.convert("RGB")
        target=torch.from_numpy(np.array(mask,dtype=np.int64,copy=True))
        result=run_adaptive_zoom(image,k=s["k"],crop_size=s["crop_size"],stride=s["stride"],
                                 overlap_threshold=s["overlap_iou"],minimum_entropy=s["minimum_entropy"])
        data=save_and_show_demo(image,result,output/f"index_{index}",target=target,
                               source_name=f"VOC index {index} | {Path(raw_val_dataset.images[index]).stem}")
        base=data["metrics"]["baseline"]["image_miou"]
        final=data["metrics"]["final"]["image_miou"]
        record={"image_index":index,"baseline":base,"final":final,"reference_rounded":expected}
        report["records"].append(record)
        assert abs(base-expected[0])<1e-4 and abs(final-expected[1])<1e-4, f"Referans eşleşmedi: {record}"
        record["status"]="PASS"
        print(f"Index {index} PASS | {base:.4f} -> {final:.4f}")
    report["status"]="PASS"
except Exception:
    import traceback
    report["status"]="FAIL"
    report["traceback"]=traceback.format_exc()
    print(report["traceback"])
finally:
    (output/"batch_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    package=shutil.make_archive(str(output),"zip",root_dir=str(output.parent),base_dir=output.name)
    print("Batch status:",report["status"])
    print("PC'ye indirilecek:",package)
    files.download(package)
