"""Configuration-file loading helpers for JSON and optional YAML."""
import json
from pathlib import Path


def load_config(path):
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install PyYAML to load YAML configuration files.") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config extension: {path.suffix}")
