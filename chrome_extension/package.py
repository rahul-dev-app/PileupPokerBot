"""Build chrome_extension/dist/pokergrid-4x4-<version>.zip from chrome_extension/extension/.

manifest.json is at the root of the zip, as the Chrome Web Store requires.
"""
import json
import zipfile
from pathlib import Path

here = Path(__file__).parent
src = here / "extension"
version = json.loads((src / "manifest.json").read_text())["version"]
out = here / "dist" / f"pokergrid-4x4-{version}.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(src.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(src))
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
