"""Audit final handoff files; optionally back up committed sources and raw evidence.

Run from the project: python Notebooks/verify_final_delivery.py [--backup]
No inference, training, or timed human rehearsal is performed.
"""
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sha = lambda data: hashlib.sha256(data).hexdigest()
def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)

def audit():
    config = tomllib.loads((ROOT/"Config/DINOv3_Final_Config.toml").read_text(encoding="utf-8"))
    assert [config["selector"][k] for k in ["k","crop_size","stride","minimum_entropy","overlap_iou","padding"]] == [3,112,28,0,.3,0]
    with zipfile.ZipFile(ROOT/config["checkpoint"]["archive"]) as z:
        assert sha(z.read(config["checkpoint"]["member"])) == config["checkpoint"]["sha256"]
    quality=read("Reproducibility/metrics_verification.json")["pooled_miou"]
    figure=read("Deliverables/Quality_Cost/figure_data.json")
    assert sha((ROOT/figure["quality_source"]).read_bytes()) == figure["quality_source_sha256"]
    mapping={"Baseline":"baseline","Sliding-window":"sliding","Adaptive K=1":"adaptive_k1",
             "Adaptive K=3":"adaptive_k3","Random K=3 three-seed mean":"random_k3_mean",
             **{f"Random {s} K=3":f"random_{s}_k3" for s in [42,43,44]}}
    with (ROOT/"Deliverables/DINOv3_Method_Comparison.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8 and {r["Method"] for r in rows}==set(mapping)
    for row in rows:
        name=mapping[row["Method"]]; timing=figure["timing_100"][name]
        assert row["mIoU"]==f"{quality[name]:.4f}"
        assert row["Delta mIoU"]==f"{(quality[name]-quality['baseline'])*100:.2f}"
        assert row["Inference Time"]==f"{timing['mean_ms']:.1f}"
        assert row["Relative Cost"]==f"{timing['mean_ms']/figure['timing_100']['baseline']['mean_ms']:.4f}"
        assert int(row["Extra Inference"])==timing["passes"]-1
        assert int(row["K"])==(0 if name=="baseline" else 25 if name=="sliding" else timing["passes"]-1)
        assert row["Crop Size"]==("n.a." if name=="baseline" else "112")
    docs=["README.md","Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md",
          "Docs/DINOv3_Final_Deney_Raporu_TR.md","Deliverables/DINOv3_Final_Demo_Akisi_TR.md",
          "Deliverables/DINOv3_Method_Comparison_Aciklama_TR.md","Deliverables/Final_Config_Decision_TR.md",
          "Notebooks/Final_Demo_Kullanim_TR.md","Deliverables/Final_Visuals/README_TR.md",
          "Deliverables/DINOv3_Teslim_Kontrolu_TR.md"]
    links=0
    for name in docs:
        text=(ROOT/name).read_text(encoding="utf-8")
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(("http:","https:","#")):
                continue
            target=unquote(target.strip("<>").split("#")[0])
            resolved=(ROOT/name).parent.joinpath(target).resolve()
            # This report is generated at the end of the same successful audit.
            if resolved != ROOT/"Reproducibility/final_delivery_audit_2026-10-27.json":
                assert resolved.is_file(),(name,target)
            links+=1
    for name in docs[:4]:
        text=(ROOT/name).read_text(encoding="utf-8")
        for value in ["0,7133","0,7220","0,7325"]:
            assert value in text,(name,value)
    notebooks=["Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb","Notebooks/DINOv3_Clean_Colab_Test.ipynb"]
    for name in notebooks:
        for cell in read(name)["cells"]:
            if cell["cell_type"]=="code":
                source="".join(cell["source"])
                ast.parse("\n".join("pass" if line.lstrip().startswith(("!","%")) else line for line in source.splitlines()))
    with zipfile.ZipFile(ROOT/"Deliverables/DINOv3_Savunma_Sunumu_TR.pptx") as z:
        slides=[n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$",n)]
        assert len(slides)==10
        text=" ".join(z.read(n).decode() for n in slides)
        for value in ["0,7133","0,7220","0,7325","147,5","510,3"]:
            assert value in text,value
    thesis_hash=sha((ROOT/"Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md").read_bytes())
    for name in ["thesis_ablation_audit_2026-10-20.json","thesis_methods_setup_audit_2026-10-27.json","thesis_benchmark_cost_audit_2026-10-27.json"]:
        record=read("Reproducibility/"+name)
        assert record["status"]=="PASS" and record["thesis_sha256"]==thesis_hash,name
    history=read("Reproducibility/experiment_manifest.json")
    for item in history["code"]["files"]:
        assert sha((ROOT/item["snapshot_path"]).read_bytes())==item["sha256"]
    visuals=read("Deliverables/Final_Visuals/package_manifest.json")
    for item in visuals["files"]:
        assert sha((ROOT/"Deliverables/Final_Visuals"/item["name"]).read_bytes())==item["sha256"]
    with zipfile.ZipFile(ROOT/"Deliverables/DINOv3_Final_Visuals_2026-10-20.zip") as z:
        assert z.testzip() is None
    preflight=read("Reproducibility/demo_rehearsal_preflight_2026-10-27.json")
    # Refresh preflight file identities after the CSV precision correction.
    for item in preflight["files"]:
        raw=(ROOT/item["path"]).read_bytes()
        item.update(sha256=sha(raw),size_bytes=len(raw))
    assert preflight["timed_rehearsal_status"]=="awaiting_user_run"
    (ROOT/"Reproducibility/demo_rehearsal_preflight_2026-10-27.json").write_text(json.dumps(preflight,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    files=docs+notebooks+["Config/DINOv3_Final_Config.toml","Deliverables/DINOv3_Method_Comparison.csv",
        "Deliverables/Quality_Cost/figure_data.json","Deliverables/DINOv3_Savunma_Sunumu_TR.pptx"]
    proof={"date":"2026-10-27","status":"PASS","scope":"Saved results, final documents, config and demo consistency; no new inference.",
        "checks":{"csv_full_precision_rows":8,"local_links":links,"notebooks_compile":True,"checkpoint_sha256":True,
                  "historical_code_snapshot_hashes":True,"final_visual_hashes_and_zip_crc":True,
                  "presentation_key_values":True,"thesis_audits_match_current_file":True},
        "files":[{"path":p,"sha256":sha((ROOT/p).read_bytes())} for p in files],
        "limits":["Timed spoken demo rehearsal awaits the user.","Thesis artifact is a methods/results chapter draft, not a completed university-formatted thesis.",
                  "Historical training commit and complete training ID list are unavailable.","Backup is local, not an off-site copy."]}
    (ROOT/"Reproducibility/final_delivery_audit_2026-10-27.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"PASS: final CSV 8 rows, {links} local links, notebooks, checkpoint, historical snapshots and final visuals.")

def backup():
    # Backup only after the reviewed sources have been committed; unrelated untracked files are left alone.
    assert not git("diff","HEAD","--name-only").strip(),"Commit tracked changes before backup."
    commit=git("rev-parse","HEAD").decode().strip()
    directory=ROOT/"Downloads/Final_Delivery_Backups"/datetime.now().strftime("%Y%m%dT%H%M%S")
    directory.mkdir(parents=True,exist_ok=False)
    names=git("ls-files","-z").decode().rstrip("\0").split("\0")
    evidence=["DINOv3_complete_20261008T122013_722899Z.zip",
              "benchmark_all_k3_100_20261007T194714_806084Z (1).zip",
              "classwise_iou_1399_20261007T203351_459375Z.zip",
              "DINOv3_clean_smoke_20261020T073625_596320Z.zip",
              "DINOv3_final_visuals_missing_20261020T074507_633421Z.zip",
              "custom_photo_demo_20261008T123108_196389Z.zip","DINOv3.ipynb"]
    names += ["Downloads/"+n for n in evidence]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/"Downloads").glob("*.json")]
    for n in names:
        assert (ROOT/n).is_file(),n
    # Preserve Git object history separately, then include it in the verified archive.
    bundle=directory/"DINOv3_history.bundle"
    subprocess.run(["git","bundle","create",str(bundle),"--all"],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(["git","bundle","verify",str(bundle)],cwd=ROOT,check=True,capture_output=True)
    entries=[]
    dest=directory/"DINOv3_Final_Delivery.zip"
    with zipfile.ZipFile(dest,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(set(names)):
            raw=(ROOT/name).read_bytes()
            if name.endswith(".zip"):
                with zipfile.ZipFile(ROOT/name) as source:
                    assert source.testzip() is None,name
            z.writestr("project/"+name,raw)
            entries.append({"path":"project/"+name,"sha256":sha(raw),"size_bytes":len(raw)})
        raw=bundle.read_bytes(); z.writestr(bundle.name,raw)
        entries.append({"path":bundle.name,"sha256":sha(raw),"size_bytes":len(raw)})
        manifest={"commit":commit,"files":entries,"backup_scope":"Local final source tree, Git history and selected essential raw archives; original files preserved."}
        z.writestr("BACKUP_MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr("RESTORE_TR.txt","ZIP'i yeni bir klasöre aç. project klasörü teslim dosyalarını ve Downloads içindeki gerekli ham arşivleri içerir. README'deki Colab sırasını izle. Git geçmişi için: git clone DINOv3_history.bundle restored_repo. Manifest her dosyanın SHA-256 değerini ve son commit'i kaydeder. Bu yerel yedeği ayrıca harici disk veya bulut depolamaya kopyala.\n")
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for e in entries:
            assert sha(z.read(e["path"]))==e["sha256"],e["path"]
    result={"status":"PASS","commit":commit,"archive":str(dest),"archive_sha256":sha(dest.read_bytes()),
            "files_verified":len(entries),"archive_bytes":dest.stat().st_size,"offsite_copy_created":False}
    (directory/"BACKUP_VERIFIED.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    if "--backup" in sys.argv:
        backup()
    else:
        audit()
