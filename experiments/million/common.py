"""Local storage adapter for the byte-identical scientific training modules."""
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def download(pin, target):
    source = Path(pin["local"])
    if sha(source) != pin["sha256"]:
        raise ValueError("Local checkpoint SHA-256 mismatch")
    shutil.copyfile(source, target)

def publish(*args, **kwargs):
    raise ValueError("This public reproduction uses local files; omit remote_prefix")

def publish_json(*args, **kwargs):
    raise ValueError("This public reproduction uses local files; omit remote_prefix")
