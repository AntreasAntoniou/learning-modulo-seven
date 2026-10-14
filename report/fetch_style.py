"""Fetch the unmodified official NeurIPS style into an ignored build directory."""
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip"

def main():
    root = Path(__file__).resolve().parent
    pin = (root / "style.sha256").read_text().split()[0]
    target = root / ".style/neurips_2026.sty"
    if not target.exists():
        data = urllib.request.urlopen(URL, timeout=60).read()
        style = zipfile.ZipFile(io.BytesIO(data)).read("neurips_2026.sty")
        if hashlib.sha256(style).hexdigest() != pin:
            raise ValueError("Official style differs from the version used for this report")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(style)
    if hashlib.sha256(target.read_bytes()).hexdigest() != pin:
        raise ValueError("Style hash differs")

if __name__ == "__main__":
    main()
