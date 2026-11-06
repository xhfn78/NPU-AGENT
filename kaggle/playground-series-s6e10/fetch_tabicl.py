"""공식 TabICLv2 체크포인트를 프로젝트 내부에 내려받습니다."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "data" / "tfm_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download


def main():
    folder = ROOT / "data" / "tabicl_model"
    folder.mkdir(parents=True, exist_ok=True)
    file = hf_hub_download(repo_id="jingang/TabICL", filename="tabicl-classifier-v2-20260212.ckpt", local_dir=folder)
    path = Path(file)
    metadata = {"repo_id": "jingang/TabICL", "filename": path.name, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("CHECKPOINT READY", metadata, flush=True)


if __name__ == "__main__":
    main()
