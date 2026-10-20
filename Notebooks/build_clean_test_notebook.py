"""Derive a fresh-runtime two-example test from the final self-contained notebook."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT/"DINOv3_Adaptive_Zoom_Final_Demo.ipynb").read_text(encoding="utf-8"))
source = (ROOT/"colab_clean_smoke_test.py").read_text(encoding="utf-8")
nb["cells"][0]["source"] = ["# DINOv3 temiz Colab testi\n", "Yeni bir GPU runtime aç. Hücreleri yukarıdan aşağıya sırayla çalıştır.\n",
    "1 → 1b config → 2 checkpoint → 3 model → 4 VOC → 5 yardımcılar → 6 iki örnek testi.\n",
    "Son hücre index 9 ve 10'u çalıştırır; ortam, config ve çıktıları ZIP indirir. Başarısızlıkta hata raporu kaydeder.\n",
    "Checkpoint içeren kendi deney ZIP'in ve Hugging Face model erişimin gerekir. Eğitim yapılmaz.\n"]
for cell in nb["cells"]:
    if cell.get("id") == "demo-title":
        cell["source"] = ["## 6 — Index 9 ve 10 temiz oturum kontrolü\n", "Varsayılan K=3 config ile iki örneği çalıştırır, referans skorları kontrol eder ve ZIP indirir.\n"]
    if cell.get("id") == "demo-run":
        cell["id"] = "clean-smoke-run"
        evidence={"final_notebook_sha256":hashlib.sha256((ROOT/"DINOv3_Adaptive_Zoom_Final_Demo.ipynb").read_bytes()).hexdigest(),
                  "pipeline_cells_sha256":{c["id"]:hashlib.sha256("".join(c["source"]).encode()).hexdigest()
                                            for c in nb["cells"] if c["cell_type"]=="code" and c["id"]!="clean-smoke-run"}}
        # Record build-time source identity without claiming an original experiment Git commit.
        source=source.replace('"stage":"preflight","examples":[],',f'"stage":"preflight","examples":[],"source_identity":{evidence!r},')
        cell["source"] = source.splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"]=None
        cell["outputs"]=[]
        python_source="\n".join("pass" if l.startswith("%pip ") else l for l in "".join(cell["source"]).splitlines())
        compile(python_source,cell["id"],"exec")
nb["metadata"]["colab"]["name"]="DINOv3_Clean_Colab_Test.ipynb"
(ROOT/"DINOv3_Clean_Colab_Test.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding="utf-8")
print("Clean test notebook generated; all code cells compiled. GPU execution pending.")
