#!/usr/bin/env python3
"""
record_model_hashes.py  —  Task 1.2 §6.4
=========================================
Writes ``model_hashes.txt``: the exact pre-trained artifact behind every
model the benchmark loads, taken from ``model_zoo/specs.py``.

For each model it records the timm HF Hub repo (``timm/<timm_id>``), the
pinned commit, the weight file, its size and its SHA-256. By default the
SHA-256 is the LFS hash published by the Hub (no download). With
``--download`` every file is fetched and hashed locally, and the local hash
must equal the published one.

Previous versions hashed HuggingFace *Transformers* repos (e.g.
google/vit-base-patch16-224-in21k, microsoft/beit-base-patch16-224-pt22k) and
a torch.hub DINO model — different artifacts from the timm weights the code
actually loads. This version hashes what is loaded.

Usage
-----
    python scripts/record_model_hashes.py [--out model_hashes.txt] [--download]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_zoo.specs import MODEL_SPECS  # noqa: E402

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def published_weights(repo: str) -> dict:
    """Commit, file name, size and LFS SHA-256 of the repo's weight file."""
    from huggingface_hub import model_info

    info = model_info(repo, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    for name in WEIGHT_FILES:
        if name in files:
            s = files[name]
            return {"repo": repo, "commit": info.sha, "file": name, "size": s.size,
                    "sha256": s.lfs.sha256 if s.lfs else None}
    raise RuntimeError(f"{repo}: no weight file among {WEIGHT_FILES}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="model_hashes.txt")
    p.add_argument("--download", action="store_true",
                   help="Download each file and verify its local SHA-256.")
    args = p.parse_args(argv)

    sep = "=" * 72
    lines = [
        "ViT Explainability Benchmark — Pre-trained Model Hash Log",
        f"Generated : {datetime.datetime.now().isoformat(timespec='seconds')}",
        "Source    : model_zoo/specs.py (the timm identifiers the code loads)",
        f"Mode      : {'downloaded + locally hashed' if args.download else 'published HF Hub LFS SHA-256 (not downloaded)'}",
        sep, "",
    ]
    failed = False
    for spec in MODEL_SPECS.values():
        rec = published_weights(spec.hf_repo)
        local = ""
        if args.download:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(spec.hf_repo, rec["file"], revision=rec["commit"])
            digest = sha256_file(path)
            ok = digest == rec["sha256"]
            failed |= not ok
            local = f"Local SHA  : {digest}  ({'MATCH' if ok else 'MISMATCH'})\n"
        lines.append(
            f"Model      : {spec.display}  [{spec.name}]{'  (ablation A4 only)' if not spec.main_matrix else ''}\n"
            f"timm ID    : {spec.timm_id}\n"
            f"Repo       : {rec['repo']}\n"
            f"Commit     : {rec['commit']}\n"
            f"File       : {rec['file']}  ({rec['size']:,} bytes)\n"
            f"SHA-256    : {rec['sha256']}\n"
            f"{local}"
            f"Pretraining: {spec.pretraining}\n"
            f"Reference  : {spec.reference}\n"
            f"{sep}\n"
        )
        print(f"  {spec.name:<14} {rec['sha256']}")
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
