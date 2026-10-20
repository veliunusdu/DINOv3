"""Execute as the final cell of the clean-test Colab notebook, after setup/data/helpers."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

smoke_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
smoke_dir = Path("/content") / f"DINOv3_clean_smoke_{smoke_stamp}"
smoke_dir.mkdir(parents=True, exist_ok=False)
smoke_report = {"created_utc":smoke_stamp,"status":"running","stage":"preflight","examples":[],
                "fresh_runtime":"User must start this notebook in a new GPU runtime; freshness is not automatically certified.",
                "scope":"Two reference demos only; no training, full evaluation, or timing benchmark."}
try:
    required = ["final_config","CONFIG_PATH","restored_model","processor","device","run_dir",
                "checkpoint_path","raw_val_dataset","run_adaptive_zoom","save_and_show_demo","image_miou"]
    missing = [name for name in required if name not in globals()]
    assert not missing, f"Önce 1, 1b, 2, 3, 4 ve 5 hücrelerini sırayla çalıştır: {missing}"
    import torch
    import numpy as np
    assert torch.cuda.is_available(), "GPU bağlantısı yok."
    active = load_final_config(CONFIG_PATH)
    selector = active["selector"]
    assert [selector[k] for k in ["k","crop_size","stride","minimum_entropy","overlap_iou","padding"]] == [3,112,28,0,.30,0]
    checkpoint_hash = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    assert checkpoint_hash == active["checkpoint"]["sha256"]
    assert model_name == active["model"]["id"] and revision == active["model"]["revision"]
    smoke_report["checkpoint_sha256"] = checkpoint_hash
    smoke_report["model"] = active["model"]
    smoke_report["config_sha256"] = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    smoke_report["environment"] = {"python":platform.python_version(),"torch":str(torch.__version__),
        "cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0),
        "packages":{p:importlib.metadata.version(p) for p in ["transformers","huggingface_hub","torchvision","numpy","Pillow","matplotlib"]}}
    (smoke_dir/"DINOv3_Final_Config.toml").write_bytes(CONFIG_PATH.read_bytes())
    assert len(raw_val_dataset) == 1449
    references = {9:{"baseline":.83006642774669,"final":.9512478779867622},
                  10:{"baseline":.4658147988040741,"final":.32078564141285404}}
    for image_index, expected in references.items():
        smoke_report["stage"] = f"inference_index_{image_index}"
        image, mask = raw_val_dataset[image_index]
        image = image.convert("RGB")
        target = torch.from_numpy(np.array(mask,dtype=np.int64,copy=True))
        result = run_adaptive_zoom(image,k=selector["k"],crop_size=selector["crop_size"],stride=selector["stride"],
                                  overlap_threshold=selector["overlap_iou"],minimum_entropy=selector["minimum_entropy"])
        scores = {name:image_miou(result[name],target) for name in ["baseline","final"]}
        record = {"image_index":image_index,"image_id":Path(raw_val_dataset.images[image_index]).stem,
                  "scores":scores,"expected":expected,"tolerance":1e-4,"selected_k":len(result["regions"])}
        smoke_report["examples"].append(record)
        assert len(result["regions"]) == 3
        assert all(abs(scores[name]-expected[name]) < 1e-4 for name in scores), f"Referans eşleşmedi: {record}"
        smoke_report["stage"] = f"export_index_{image_index}"
        example_dir = smoke_dir / f"index_{image_index}"
        save_and_show_demo(image,result,example_dir,target=target,source_name=f"VOC index {image_index}")
        record["output_files"] = {p.relative_to(smoke_dir).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(example_dir.rglob("*")) if p.is_file()}
        assert record["output_files"], "Örnek çıktıları oluşmadı."
        record["status"] = "PASS"
        print(f"Index {image_index} PASS: {scores}")
    smoke_report["status"] = "PASS"
    smoke_report["stage"] = "complete"
except Exception:
    smoke_report["status"] = "FAIL"
    smoke_report["traceback"] = traceback.format_exc()
    print(smoke_report["traceback"])
finally:
    (smoke_dir/"smoke_report.json").write_text(json.dumps(smoke_report,ensure_ascii=False,indent=2),encoding="utf-8")
    smoke_zip = shutil.make_archive(str(smoke_dir),"zip",root_dir=str(smoke_dir.parent),base_dir=smoke_dir.name)
    print("Test durumu:",smoke_report["status"])
    print("Kanıt veya hata paketi:",smoke_zip)
    from google.colab import files
    files.download(smoke_zip)
