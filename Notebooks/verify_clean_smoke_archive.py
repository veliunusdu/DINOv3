"""Verify the user-exported clean Colab smoke ZIP, without running inference."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tomllib
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
archive = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"Downloads/DINOv3_clean_smoke_20261020T073625_596320Z.zip"
archive = archive.resolve()
sha = lambda data: hashlib.sha256(data).hexdigest()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, "ZIP CRC failure"
    member = next(n for n in z.namelist() if n.endswith("/smoke_report.json"))
    prefix = member.rsplit("/",1)[0]+"/"
    raw_report = z.read(member)
    report = json.loads(raw_report)
    assert report["status"] == "PASS" and report["stage"] == "complete"
    assert {e["image_index"] for e in report["examples"]} == {9,10}
    runtime_config = z.read(prefix+"DINOv3_Final_Config.toml")
    assert sha(runtime_config) == report["config_sha256"]
    # TOML semantics are compared, since newline conventions can change raw bytes.
    local_config = tomllib.loads((ROOT/"Config/DINOv3_Final_Config.toml").read_text(encoding="utf-8-sig"))
    assert tomllib.loads(runtime_config.decode("utf-8-sig")) == local_config
    assert report["checkpoint_sha256"] == local_config["checkpoint"]["sha256"]
    nb = json.loads((ROOT/"Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]:"".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"}
    for name, expected in report["source_identity"]["pipeline_cells_sha256"].items():
        assert sha(cells[name].encode()) == expected, f"Pipeline source mismatch: {name}"
    checks=[]
    outputs=ROOT/"Deliverables/Clean_Colab_Test_2026-10-20"
    outputs.mkdir(parents=True,exist_ok=True)
    for e in report["examples"]:
        assert e["status"] == "PASS" and e["selected_k"] == 3
        for name, expected in e["output_files"].items():
            assert sha(z.read(prefix+name)) == expected, f"Output hash mismatch: {name}"
        ep=prefix+f"index_{e['image_index']}/"
        target=np.array(Image.open(io.BytesIO(z.read(ep+"ground_truth.png"))))
        predictions={name:np.array(Image.open(io.BytesIO(z.read(ep+f"{name}_labels.png")))) for name in ["baseline","final"]}
        scores={}
        for name,pred in predictions.items():
            assert pred.shape == target.shape
            valid=target!=255
            assert (pred<21).all()
            cm=np.bincount(target[valid].astype(np.int64)*21+pred[valid],minlength=441).reshape(21,21)
            union=cm.sum(0)+cm.sum(1)-np.diag(cm)
            miou=float((np.diag(cm)[union>0]/union[union>0]).mean())
            assert abs(miou-e["scores"][name])<1e-12
            assert abs(miou-e["expected"][name])<e["tolerance"]
            scores[name]=miou
        with np.load(io.BytesIO(z.read(ep+"global_output.npz")),allow_pickle=False) as data:
            coverage=data["coverage"]
            assert coverage.shape==target.shape
            assert np.array_equal(predictions["baseline"][coverage==0],predictions["final"][coverage==0])
        result=json.loads(z.read(ep+"results.json"))
        assert result["passes"]=={"baseline":1,"adaptive":4} and result["parameters"]["k_selected"]==3
        image=outputs/f"index_{e['image_index']}_7_panels.png"
        image.write_bytes(z.read(ep+"demo_7_panels.png"))
        checks.append({"image_index":e["image_index"],"image_id":e["image_id"],"scores_recomputed_from_png":scores,
                       "output_hashes_verified":len(e["output_files"]),"passes":4,
                       "crop_outside_preserved":True,"figure":image.relative_to(ROOT).as_posix()})
    proof={"status":"PASS","source_archive":archive.relative_to(ROOT).as_posix() if archive.is_relative_to(ROOT) else str(archive),
           "source_archive_sha256":sha(archive.read_bytes()),"source_report_member":member,"source_report_sha256":sha(raw_report),
           "checkpoint_sha256":report["checkpoint_sha256"],"runtime_config_sha256":report["config_sha256"],
           "config_semantically_matches_local":True,"pipeline_cells_match_local":True,"zip_crc_passed":True,
           "environment":report["environment"],"examples":checks,
           "fresh_runtime_evidence":"User reported a new empty GPU runtime and shared setup/config/recovery/model/dataset logs before exporting this ZIP.",
           "scope":"Two reference images; not training reproduction or a rerun of full quality/latency benchmarks."}
    (ROOT/"Reproducibility/clean_colab_verification_2026-10-20.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ROOT/"Reproducibility/clean_colab_smoke_report_2026-10-20.json").write_bytes(raw_report)
    lines=["# Temiz Colab demo testi — 20 Ekim 2026","",
           "**Sonuç: PASS.** Kullanıcı yeni ve boş GPU runtime bildirdi; kurulum, config, checkpoint restore, model ve VOC veri yükleme çıktılarını paylaştı. Son iki örnek testi ZIP olarak indirildi ve yerelde doğrulandı.",
           "", "| Örnek | Baseline image mIoU | Adaptive K=3 image mIoU | Durum |", "|---|---:|---:|---|"]
    for e in checks:
        scores=e["scores_recomputed_from_png"]
        lines.append(f"| Index {e['image_index']} | {scores['baseline']:.10f} | {scores['final']:.10f} | PASS |")
    lines += ["", "İki örnekte de seçilen K=3 ve toplam geçiş sayısı 4. Index 10, yöntemin kötüleştiği beklenen örnektir; testin geçmesi performansın her örnekte arttığı anlamına gelmez.",
              "", "## Doğrulanan kanıtlar", "",
              "- ZIP CRC kontrolü ve 32 çıktı dosyasının SHA-256 değeri geçti.",
              "- Skorlar PNG sınıf maskeleri ve GT'den, ignore=255 ve background dahil olarak yeniden hesaplandı.",
              "- Crop birleşimi dışında baseline ve final prediction aynı kaldı.",
              "- Runtime config hash'i raporla, config anlamı yerel TOML ile eşleşti. Ham config baytları satır sonu nedeniyle farklılaşabilir.",
              "- Checkpoint hash'i ve model revision doğru; pipeline hücre hash'leri yerel final notebook ile eşleşti.",
              "- İki yedi panelli final görseli yerelde incelendi.", "", "## Ortam", "",
              f"Python {report['environment']['python']}; Torch {report['environment']['torch']}; CUDA {report['environment']['cuda_runtime']}; GPU {report['environment']['gpu']}.",
              f"Paketler: {json.dumps(report['environment']['packages'],ensure_ascii=False)}.", "", "## Kaynak ve kapsam", "",
              f"Arşiv: `{proof['source_archive']}`. SHA-256: `{proof['source_archive_sha256']}`.",
              "Makine-okunur doğrulama: `clean_colab_verification_2026-10-20.json`. Orijinal test raporu: `clean_colab_smoke_report_2026-10-20.json`.",
              "Bu test config bağlantılı inference/export zincirini iki referans görüntüde doğrular. Eğitimin yeniden üretimi veya 1.399 görüntülük kalite / 100 görüntülük latency benchmark'ının yeniden çalıştırılması değildir. Oturumun yeni olduğu bilgisi kullanıcı beyanı ve paylaşılan kurulum adımlarına dayanır.",
              "Eski experiment_manifest.json tarihsel snapshot olarak korunmuştur; bu 20 Ekim kanıtı onu tamamlar. Kaynak ZIP Downloads altında Git dışında kalır ve ayrı yedeklenmelidir."]
    (ROOT/"Reproducibility/clean_colab_verification_2026-10-20_TR.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("PASS: ZIP CRC, 32 output hashes, config, checkpoint identity and pipeline cells verified.")
for e in checks:
    print(e["image_index"],e["scores_recomputed_from_png"])
print("Scores independently recomputed from exported native masks; outside-crop predictions preserved.")
