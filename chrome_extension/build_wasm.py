"""Build the WebAssembly SIMD kernels (chrome_extension/wasm) into chrome_extension/extension/mlp.wasm.

    python chrome_extension/build_wasm.py   # needs cargo and `rustup target add wasm32-unknown-unknown`
"""
import os
import shutil
import subprocess
from pathlib import Path

here = Path(__file__).parent
env = {**os.environ, "RUSTFLAGS": "-C target-feature=+simd128"}
subprocess.run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"], cwd=here / "wasm", env=env, check=True)
out = here / "extension" / "mlp.wasm"
shutil.copy(here / "wasm" / "target" / "wasm32-unknown-unknown" / "release" / "pp_mlp.wasm", out)
print(f"wrote {out} ({out.stat().st_size} bytes)")
