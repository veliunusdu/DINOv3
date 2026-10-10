"""Build the self-contained Colab notebook from existing recovery cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
startup = json.loads((ROOT / "DINOv3_Her_Oturumda_Baslangic.ipynb").read_text(encoding="utf-8"))
helpers = (ROOT / "adaptive_zoom_demo.py").read_text(encoding="utf-8")


def original_source(index):
    source = startup["cells"][index]["source"]
    return "".join(source) if isinstance(source, list) else source


def md(identity, source):
    return {"cell_type": "markdown", "id": identity, "metadata": {}, "source": source.splitlines(True)}


def code(identity, source):
    return {"cell_type": "code", "id": identity, "metadata": {},
            "execution_count": None, "outputs": [], "source": source.splitlines(True)}


dataset = '''from torchvision.datasets import VOCSegmentation
voc_root = Path("/content/VOC")
voc_ready = (voc_root / "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt").is_file()
raw_val_dataset = VOCSegmentation(str(voc_root), year="2012", image_set="val",
                                 download=not voc_ready)
assert len(raw_val_dataset) == 1449
print("Validation images:", len(raw_val_dataset))
'''

demo = '''# Bu hücrede parametreleri değiştirip yeniden çalıştırabilirsin.
SOURCE = "voc"          # "voc" veya "upload"
IMAGE_INDEX = 9
K = 3
CROP_SIZE = 112
STRIDE = 28
OVERLAP_THRESHOLD = 0.30

assert SOURCE in ("voc", "upload")
if SOURCE == "voc":
    assert "raw_val_dataset" in globals(), "Önce VOC veri hücresini çalıştır."
    assert 0 <= IMAGE_INDEX < len(raw_val_dataset)
    demo_image, demo_mask = raw_val_dataset[IMAGE_INDEX]
    demo_image = demo_image.convert("RGB")
    demo_target = torch.from_numpy(np.array(demo_mask, dtype=np.int64, copy=True))
    demo_name = f"VOC index {IMAGE_INDEX} | {Path(raw_val_dataset.images[IMAGE_INDEX]).stem}"
else:
    uploaded_photo = files.upload()
    assert len(uploaded_photo) == 1, "Tek bir fotoğraf seç."
    import io
    photo_name, photo_bytes = next(iter(uploaded_photo.items()))
    with Image.open(io.BytesIO(photo_bytes)) as photo:
        demo_image = ImageOps.exif_transpose(photo).convert("RGB")
    demo_target = None
    demo_name = Path(photo_name).name

demo_result = run_adaptive_zoom(
    demo_image, k=K, crop_size=CROP_SIZE, stride=STRIDE,
    overlap_threshold=OVERLAP_THRESHOLD,
)

# Sabit referans örneğinde önceki deneyle eşleşme kontrolü.
if SOURCE == "voc" and IMAGE_INDEX == 9 and CROP_SIZE == 112 and STRIDE == 28 and OVERLAP_THRESHOLD == 0.30:
    expected = {1: 0.9214, 3: 0.9512, 5: 0.9597}
    assert abs(image_miou(demo_result["baseline"], demo_target) - 0.8301) < 0.0001
    if K in expected:
        assert len(demo_result["regions"]) == K
        assert abs(image_miou(demo_result["final"], demo_target) - expected[K]) < 0.0001
    print("Önceki referans görüntü sonuçlarıyla eşleşti.")

demo_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
demo_dir = Path(run_dir) / f"final_demo_{demo_id}"
demo_report = save_and_show_demo(demo_image, demo_result, demo_dir,
                                 target=demo_target, source_name=demo_name)
demo_zip = shutil.make_archive(str(demo_dir), "zip",
                               root_dir=str(demo_dir.parent), base_dir=demo_dir.name)
print("PC'ye indirilecek demo paketi:", demo_zip)
files.download(demo_zip)
'''

export = '''# Oturum sonunda checkpoint ve bütün sonuçları tek paket halinde indir.
# Bu ZIP notebook kodunu içermez; .ipynb dosyasını da ayrıca sakla.
assert Path(run_dir).is_dir()
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
full_zip = shutil.make_archive(f"/content/DINOv3_complete_{timestamp}", "zip",
                              root_dir=str(Path(run_dir).parent), base_dir=Path(run_dir).name)
print("Tam deney paketi:", full_zip)
files.download(full_zip)
'''

cells = [
    md("intro", """# DINOv3 Adaptive Zoom — final demo

Eğitilmiş checkpoint'i yükler; eğitim veya 100/1399 görüntülük değerlendirme başlatmaz.
Yeni Colab oturumunda T4 GPU seç; 1–6 hücrelerini sırayla çalıştır.
Model/veri zaten hazırsa yalnızca **5 ve 6** yeterlidir.
Mevcut deney protokolü: 224×224 input, entropy, configurable Top-K,
original-image crops, aynı modelle second pass, overlap'te local logits ortalaması.
Varsayılan K=3; K=1 veya K=5 de seçilebilir. NMS nedeniyle gerçek seçim sayısı K'dan az olabilir.
Özel fotoğrafta GT yoksa doğruluk metriği hesaplanmaz.

Doğruluk kıyaslaması: 1399 görüntüde baseline 0.7133, K=3 0.7220, sliding 0.7325.
Son ayrı 100 görüntülük T4 benchmark'ında K=3 147.5 ms, sliding 510.3 ms.
Bu tek görüntü demosu genel değerlendirmeyi veya bilimsel benchmark'ı tekrarlamaz.
"""),
    md("setup-title", "## 1 — Bağımlılıklar\nColab yeniden başlatma isterse yeniden başlatıp 1'den devam et."),
    code("setup", original_source(2)),
    md("recover-title", "## 2 — Checkpoint ve sonuç paketi\nTek bir tam deney ZIP'i seç. Galeri ZIP'i yerine checkpoint içeren paket kullan."),
    code("recover", original_source(4)),
    md("model-title", "## 3 — Backbone ve eğitilmiş head\nHugging Face hesabının DINOv3 erişimi gerekir. Token'ı paylaşma."),
    code("model", original_source(6).replace("from huggingface_hub import login", "from huggingface_hub import notebook_login").replace("\nlogin()", "\nnotebook_login()")),
    md("data-title", "## 4 — VOC validation\nVeri varsa yeniden indirmez. Özel fotoğraf demosunda bu hücre atlanabilir."),
    code("data", dataset),
    md("helpers-title", "## 5 — Selector, second pass, fusion ve görselleştirme\nGT selector'a verilmez; yalnızca sonuç metrikleri için kullanılır. Bu hücre inference başlatmaz."),
    code("demo-helpers", helpers),
    md("demo-title", "## 6 — Tek görüntülük demo\nParametreleri ayarla, çalıştır. Yedi panel ve yerel logits/koordinatlar ZIP olarak indirilir.\nPanel sırası: Original, GT, global prediction, entropy, crop kutuları, local predictions, final segmentation."),
    code("demo-run", demo),
    md("export-title", "## 7 — Oturum sonunda tam yedek (isteğe bağlı)\nNotebook'u ayrıca Dosya → İndir → .ipynb ile kaydet."),
    code("export", export),
]

notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"name": "DINOv3_Adaptive_Zoom_Final_Demo.ipynb"},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}
output = ROOT / "DINOv3_Adaptive_Zoom_Final_Demo.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")

for cell in cells:
    if cell["cell_type"] == "code":
        # IPython %pip is checked as valid Python after replacing just that line.
        source = "".join(cell["source"])
        source = "\n".join("pass" if line.startswith("%pip ") else line
                           for line in source.splitlines())
        compile(source, f"{output.name}:{cell['id']}", "exec")
assert len({cell["id"] for cell in cells}) == len(cells)

# Check the coordinate and overlap functions independently of unavailable local torch.
tree = ast.parse(helpers)
functions = ast.Module(body=[node for node in tree.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name in ("box_iou", "topk_box_to_original")], type_ignores=[])
import math
namespace = {"math": math}
exec(compile(functions, "coordinate-checks", "exec"), namespace)
assert namespace["box_iou"]([0, 0, 10, 10], [10, 0, 20, 10]) == 0
assert namespace["box_iou"]([0, 0, 10, 10], [0, 0, 10, 10]) == 1
assert namespace["topk_box_to_original"]([0, 112, 112, 224], (500, 375)) == [0, 187, 250, 375]
assert namespace["topk_box_to_original"]([0, 0, 224, 224], (333, 501)) == [0, 0, 333, 501]
print("Notebook generated:", output)
print("7 code cells compiled; cell IDs and coordinate/IoU checks passed.")
print("GPU inference and seven-panel rendering must be checked in Colab.")
