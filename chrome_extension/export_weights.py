"""Export afterstate value MLPs (afterstate.py, model "mlp") for the extension's in-browser value search.

    python chrome_extension/export_weights.py

Writes chrome_extension/extension/nets/<name>.bin (float32 little-endian: w1, b1, w2, b2, ... in order)
and nets/nets.json (layer sizes and value_scale of each model).
"""
import json
import pickle
from pathlib import Path

import numpy as np

MODELS = {
    "value": "models/afterstate_mlp/afterstate_params.pkl",  # beam values and leaves (512 wide)
    "inner": "models/td_h128/afterstate_params.pkl",  # plays the sampled next round card by card (128 wide)
}
OUT = Path(__file__).parent / "extension" / "nets"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = {}
    for name, path in MODELS.items():
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["model"] != "mlp":
            raise ValueError(f"{path}: only the mlp model is supported")
        params = payload["params"]
        num_layers = sum(k.startswith("w") for k in params)
        arrays, sizes = [], []
        for i in range(1, num_layers + 1):
            w = np.asarray(params[f"w{i}"], np.float32)
            arrays += [w.ravel(), np.asarray(params[f"b{i}"], np.float32).ravel()]
            sizes.append(w.shape[0])
        sizes.append(1)
        data = np.concatenate(arrays).astype("<f4")
        (OUT / f"{name}.bin").write_bytes(data.tobytes())
        meta[name] = {"file": f"{name}.bin", "source": path, "sizes": sizes, "value_scale": payload["value_scale"]}
        print(f"{name}: {path} -> {OUT / (name + '.bin')} ({data.size} floats, sizes {sizes})")
    (OUT / "nets.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
