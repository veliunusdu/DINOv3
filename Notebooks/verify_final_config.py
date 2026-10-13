"""CPU-only checks for config propagation; does not claim GPU inference parity."""
import ast
import copy
import json
import tempfile
from pathlib import Path
from final_config import load_final_config, validate_final_config

root = Path(__file__).resolve().parent.parent
config_path = root / "Config/DINOv3_Final_Config.toml"
config = load_final_config(config_path)
assert tuple(config["selector"][k] for k in
             ("k", "crop_size", "stride", "minimum_entropy", "overlap_iou", "padding")) == (3,112,28,0,0.30,0)
for section, key, value in [("selector","k",0),("selector","minimum_entropy",1.1),
                            ("selector","padding",10),("fusion","mode","confidence_weighted")]:
    bad = copy.deepcopy(config)
    bad[section][key] = value
    try:
        validate_final_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid config accepted: {section}.{key}")
notebook = json.loads((root / "Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb").read_text(encoding="utf-8"))
cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
with tempfile.TemporaryDirectory(dir=root / "Artifacts") as temp:
    assert Path(temp).resolve().is_relative_to((root / "Artifacts").resolve())
    path = Path(temp) / "final.toml"
    source = cells["final-config"].replace('Path("/content/DINOv3_Final_Config.toml")', f"Path({str(path)!r})")
    namespace = {}
    exec(compile(source, "config-cell", "exec"), namespace)
    assert namespace["final_config"] == config
    path.write_text(config_path.read_text(encoding="utf-8").replace("k = 3", "k = 5")
                    .replace("minimum_entropy = 0.0", "minimum_entropy = 0.2"), encoding="utf-8")
    exec(compile(source, "custom-config-cell", "exec"), namespace)
    assert namespace["final_config"]["selector"]["k"] == 5
    assert namespace["final_config"]["selector"]["minimum_entropy"] == 0.2
    # Execute the actual demo parameter assignments, without model inference.
    tree = ast.parse(cells["demo-run"])
    names = {"K", "CROP_SIZE", "STRIDE", "OVERLAP_THRESHOLD", "MINIMUM_ENTROPY"}
    assignments = ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign)
                                  and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names],type_ignores=[])
    exec(compile(assignments,"demo-bindings","exec"),namespace)
    assert (namespace["K"],namespace["MINIMUM_ENTROPY"]) == (5,0.2)
    assert namespace["CROP_SIZE"] == 112
assert "SEED = final_config" in cells["recover"]
assert "minimum_entropy=MINIMUM_ENTROPY" in cells["demo-run"]
assert 'candidate["score"] < minimum_entropy' in cells["demo-helpers"]
print("Default/custom TOML, invalid values, demo bindings and seed/threshold wiring passed.")
print("GPU inference parity must be checked in a fresh Colab runtime.")
