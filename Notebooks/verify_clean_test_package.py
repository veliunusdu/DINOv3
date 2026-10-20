"""CPU-only test of the clean notebook's error export path. No GPU inference."""
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import zipfile
from contextlib import redirect_stdout

root=Path(__file__).resolve().parent.parent
nb=json.loads((root/"Notebooks/DINOv3_Clean_Colab_Test.ipynb").read_text(encoding="utf-8"))
source=next("".join(c["source"]) for c in nb["cells"] if c["id"]=="clean-smoke-run")
with tempfile.TemporaryDirectory(dir=root/"Artifacts") as temporary:
    assert Path(temporary).resolve().is_relative_to((root/"Artifacts").resolve())
    source=source.replace('Path("/content")',f'Path({temporary!r})')
    downloads=[]
    colab=types.ModuleType("google.colab")
    colab.files=types.SimpleNamespace(download=lambda p:downloads.append(p))
    google=types.ModuleType("google")
    google.colab=colab
    with patch.dict(sys.modules,{"google":google,"google.colab":colab}),redirect_stdout(io.StringIO()):
        exec(compile(source,"mocked-preflight-test","exec"),{})
    assert len(downloads)==1
    with zipfile.ZipFile(downloads[0]) as z:
        name=next(n for n in z.namelist() if n.endswith("smoke_report.json"))
        report=json.loads(z.read(name))
    assert report["status"]=="FAIL" and report["stage"]=="preflight"
    assert "Önce" in report["traceback"]
    assert report["source_identity"]["final_notebook_sha256"]
print("Offline missing-state/error-ZIP export test passed. Live GPU test remains pending.")
