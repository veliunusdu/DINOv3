"""Choose development examples and render available raw exports; no new inference."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tomllib
import zipfile

ROOT=Path(__file__).resolve().parent.parent
deps=ROOT/"Artifacts/plot_dependencies"
if deps.exists(): sys.path.insert(0,str(deps))
cache=ROOT/"tmp/matplotlib_cache"
cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(cache))
import numpy as np
from PIL import Image
from final_visuals_render import render_final_panels,render_oracle_reference

OUT=ROOT/"Deliverables/Final_Visuals"
OUT.mkdir(parents=True,exist_ok=True)
main=ROOT/"Downloads/DINOv3_complete_20261008T122013_722899Z.zip"
prefix="20261001T181030_523202Z/"
score_member=prefix+"topk_comparison_50_20261006T223417_739370Z/results.json"
gallery_member=prefix+"selector_gallery_20_20261006T221453_694935Z/gallery_metadata.json"
with zipfile.ZipFile(main) as z:
    score_bytes=z.read(score_member)
    records={r["image_index"]:r for r in json.loads(score_bytes)["records"]}
    gallery=json.loads(z.read(gallery_member))
    gallery_sha256=hashlib.sha256(z.read(gallery_member)).hexdigest()
selected=[(9,"success","Şişe/kişi sınırlarında iyileşme; temiz test ham çıktıları mevcut."),
          (13,"success","Tekne hedefi: farklı nesne sınıfında belirgin iyileşme."),
          (18,"success","Bitki hedefi: Top-K ile ilk seçimin dışındaki bölgelere erişim."),
          (23,"success","Binici/kişi hedefi: farklı sahnede iyileşme."),
          (10,"failure","Sofa bölgesinde yerel sınıf parçalanması ve yeni yanlış pozitifler."),
          (14,"failure","Monitör içeriği ile nesne sınıflarının karıştığı yerel refinement." )]
manifest={"selection_scope":"Six deliberately selected development examples, not an unbiased evaluation sample.",
          "selection_rule":"Four improving examples across target classes and two diagnosed failures; not top-gain ranking.",
          "score_archive":main.relative_to(ROOT).as_posix(),"score_member":score_member,
          "score_member_sha256":hashlib.sha256(score_bytes).hexdigest(),"examples":[]}
for index,category,reason in selected:
    r=records[index]
    base=r["methods"]["baseline"]["image_miou"]
    final=r["methods"]["adaptive_k3"]["image_miou"]
    assert (final>base) if category=="success" else (final<base)
    manifest["examples"].append({"image_index":index,"image_id":r["image_id"],"category":category,"reason_tr":reason,
       "reference_scores":{"baseline":base,"final":final},"delta_points":(final-base)*100,"expected_k":3,"expected_passes":4})

archives=[ROOT/"Downloads/DINOv3_clean_smoke_20261020T073625_596320Z.zip"]
archives+=sorted((ROOT/"Downloads").glob("DINOv3_final_visuals_missing_*.zip"))
rendered=[]
archive_checks=[]
local_config=tomllib.loads((ROOT/"Config/DINOv3_Final_Config.toml").read_text(encoding="utf-8-sig"))
for archive in archives:
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        reports=[n for n in z.namelist() if n.endswith(("batch_report.json","smoke_report.json"))]
        assert len(reports)==1
        report=json.loads(z.read(reports[0]))
        assert report["status"]=="PASS"
        assert report["checkpoint_sha256"]==local_config["checkpoint"]["sha256"]
        config_member=reports[0].rsplit("/",1)[0]+"/DINOv3_Final_Config.toml"
        config_bytes=z.read(config_member)
        assert hashlib.sha256(config_bytes).hexdigest()==report["config_sha256"]
        assert tomllib.loads(config_bytes.decode("utf-8-sig"))==local_config
        archive_checks.append({"archive":archive.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                               "crc":"PASS","report_status":report["status"],"checkpoint_config_match":True})
        for e in manifest["examples"]:
            index=e["image_index"]
            if index in rendered: continue
            matches=[n for n in z.namelist() if n.endswith(f"index_{index}/results.json")]
            if not matches: continue
            folder=matches[0].rsplit("/",1)[0]+"/"
            result=json.loads(z.read(matches[0]))
            scores={name:result["metrics"][key]["image_miou"] for name,key in [("baseline","baseline"),("final","final")]}
            for name,value in scores.items(): assert abs(value-e["reference_scores"][name])<1e-4
            def img(name):
                p=Image.open(io.BytesIO(z.read(folder+name)));p.load();return p
            image=img("original.png").convert("RGB")
            target=np.array(img("ground_truth.png"))
            baseline=np.array(img("baseline_labels.png"))
            final=np.array(img("final_labels.png"))
            assert baseline.shape==final.shape==target.shape==(image.height,image.width)
            valid=target!=255
            assert valid.any() and (target[valid]<21).all()
            for name,pred in [("baseline",baseline),("final",final)]:
                assert (pred<21).all()
                cm=np.bincount(target[valid].astype(np.int64)*21+pred[valid],minlength=441).reshape(21,21)
                union=cm.sum(0)+cm.sum(1)-np.diag(cm)
                recomputed=float(np.mean(np.diag(cm)[union>0]/union[union>0]))
                assert abs(recomputed-scores[name])<1e-12
            assert result["passes"]=={"baseline":1,"adaptive":4} and len(result["regions"])==3
            with np.load(io.BytesIO(z.read(folder+"global_output.npz")),allow_pickle=False) as state:
                entropy=state["entropy"].copy()
                coverage=state["coverage"].copy()
                assert entropy.shape==(224,224) and np.isfinite(entropy).all()
                assert ((entropy>=0)&(entropy<=1)).all()
            assert np.array_equal(baseline[coverage==0],final[coverage==0])
            locals=[]
            for region in result["regions"]:
                x1,y1,x2,y2=region["original_box"]
                assert 0<=x1<x2<=image.width and 0<=y1<y2<=image.height
                with np.load(io.BytesIO(z.read(folder+f"crop_{region['rank']:02d}_output.npz")),allow_pickle=False) as crop:
                    pred=crop["prediction_native"].copy()
                    assert pred.shape==(y2-y1,x2-x1)
                    assert crop["original_box"].tolist()==region["original_box"]
                    locals.append(pred)
            output=OUT/f"{e['category']}_index_{index}_8_panels.png"
            render_final_panels(image,target,baseline,entropy,result["regions"],img("local_predictions.png"),final,
                                output,f"VOC index {index} | {e['image_id']}",scores,local_predictions=locals)
            e["figure"]=output.relative_to(ROOT).as_posix()
            e["source_archive"]=archive.relative_to(ROOT).as_posix()
            e["source_archive_sha256"]=hashlib.sha256(archive.read_bytes()).hexdigest()
            e["figure_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
            e["checks"]={"native_mask_scores_recomputed":True,"outside_crop_preserved":True,"crop_coordinates_match":True,
                          "baseline_wrong_pixels":int(((baseline!=target)&valid).sum()),
                          "final_wrong_pixels":int(((final!=target)&valid).sum()),
                          "corrected_pixels":int(((baseline!=target)&(final==target)&valid).sum()),
                          "new_wrong_pixels":int(((baseline==target)&(final!=target)&valid).sum())}
            rendered.append(index)
            if index==9:
                oracle=next(r for r in gallery["records"] if r["image_index"]==9)
                render_oracle_reference(image,target,entropy,result["regions"][0]["original_box"],
                                       oracle["metrics"]["oracle_gt_padded"]["box"],oracle["target_class"],OUT/"oracle_vs_adaptive_index_9.png")
                manifest["oracle_reference"]={"image_index":9,"target_class":oracle["target_class"],
                    "adaptive_k1_box":result["regions"][0]["original_box"],"oracle_gt_box":oracle["metrics"]["oracle_gt_padded"]["box"],
                    "source_member":gallery_member,"source_member_sha256":gallery_sha256,
                    "comparison_scope":"Selection boxes only; variable area, GT-assisted oracle; not a matched-budget accuracy comparison."}
manifest["rendered_indices"]=rendered
manifest["pending_indices"]=[e["image_index"] for e in manifest["examples"] if e["image_index"] not in rendered]
manifest["status"]="complete" if not manifest["pending_indices"] else "partial"
manifest["archive_checks"]=archive_checks
(OUT/"selection_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
lines=["# Final görsel seçimi ve kapsamı","", "Dört başarı ve iki failure örneği development grubundan amaçlı seçildi; bağımsız test veya ortalama başarı kanıtı değildir.",
       "", "Sekiz panel: input, GT, baseline, baseline error map, normalize entropy, Top-K kutuları, second-pass çıktıları ve final fusion. Error map beyaz=geçerli hatalı piksel, siyah=doğru, gri=ignore=255.",
       "", "| Index | Görüntü ID | Tür | Baseline | K=3 | Fark (puan) | Durum |", "|---|---|---|---:|---:|---:|---|"]
for e in manifest["examples"]:
    s=e["reference_scores"]
    lines.append(f"| {e['image_index']} | {e['image_id']} | {e['category']} | {s['baseline']:.4f} | {s['final']:.4f} | {e['delta_points']:+.2f} | {'Görsel hazır' if e['image_index'] in rendered else 'Ham çıktı bekliyor'} |")
lines += ["", "Oracle–Adaptive görseli index 9 için K=1 entropy kutusu ve önceki galeride kayıtlı GT hedef kutusunu gösterir. Oracle GT kullanır ve crop alanı farklıdır; eş bütçeli doğruluk kıyası veya üst sınır iddiası değildir.",
          "", "9 ve 10'un ham çıktıları temiz Colab testinden yeniden kullanıldı. Eksik dört örnek aynı oturumda ayrıca işlendi. Altı örneğin skorları PNG maskelerinden yeniden hesaplandı; checkpoint/config, ZIP CRC, crop koordinatları ve crop dışının korunması kontrol edildi. Kaynaklar ve hash'ler selection_manifest.json içinde.",
          "", "## Tez için şekil açıklamaları", ""]
captions={9:"Şişe ve kişi içeren sahnede seçilen üç bölgeyle görüntü mIoU artmıştır. Nesne sınırlarında iyileşme görülebilir; bu tek örnek sınıf genelinde üstünlük kanıtı değildir.",
          13:"Tekne sahnesinde baseline ve K=3 çıktıları karşılaştırılır. Hedef sınıfı tekne olan bu development örneğinde görüntü mIoU artmıştır; kazanç yalnızca küçük nesnelere atfedilmez.",
          18:"Bitki hedefi içeren iç mekân sahnesinde K=3 sonucu baseline'a göre daha yüksek görüntü mIoU vermiştir. Şekil sınıf veya nesne boyutu bazında genel başarı iddiası taşımaz.",
          23:"At ve binici içeren sahnede global tahmin, entropy seçimi ve yerel tahminler gösterilir. Fusion sonrasında görüntü mIoU yükselmiştir.",
          10:"Yüksek entropy'li sofa alanına yönelen crop'lar, yerel tahminde sınıf parçalanması üretmiştir. Görüntüde kedi görünmesine rağmen geçerli GT'de cat sınıfı bulunmadığından ilgili tahmin yanlış pozitif olarak değerlendirilir. Ölçek/bağlam etkisi olasıdır, bu şekille nedensel olarak ayrıştırılmaz.",
          14:"Monitör içeriği ve sahnedeki nesne sınıfları arasındaki karışıklık, yerel tahminin fusion ile aktarılmasından sonra görüntü mIoU kaybına eşlik eder. Gerçek GT şişesinin seçilmesi garanti değildir; değerlendirme GT anotasyonuna göre yapılır."}
for e in manifest["examples"]:
    s=e["reference_scores"]
    lines += [f"### Index {e['image_index']} — {e['image_id']}","",captions[e["image_index"]],"",
              f"Native image mIoU: {s['baseline']:.4f} → {s['final']:.4f}; fark {e['delta_points']:+.2f} yüzde puan. K=3, toplam dört geçiş. Ignore=255 gri gösterilir.",""]
    if "figure" in e:
        lines += [f"![Index {e['image_index']} sekiz panelli sonuç]({Path(e['figure']).name})", ""]
(OUT/"README_TR.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if manifest["status"]=="complete":
    files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="package_manifest.json"]
    inventory={"scope":"Six eight-panel scientific figures, one oracle reference, captions and source provenance.",
               "files":[{"name":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size} for p in files],
               "plotting_versions":{"numpy":np.__version__},"source_archives_required_for_regeneration":archive_checks}
    (OUT/"package_manifest.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    package=ROOT/"Deliverables/DINOv3_Final_Visuals_2026-10-20.zip"
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file(): z.write(p,"Final_Visuals/"+p.name)
        for name in ["final_visuals_render.py","prepare_final_visuals.py","colab_missing_final_visuals.py"]:
            z.write(ROOT/"Notebooks"/name,"reproduction_sources/"+name)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for item in inventory["files"]:
            assert hashlib.sha256(z.read("Final_Visuals/"+item["name"])).hexdigest()==item["sha256"]
    print("Verified package:",package,"bytes:",package.stat().st_size)
print("Rendered:",rendered,"Pending:",manifest["pending_indices"])
