#!/usr/bin/env python3
"""Fetch the Mondk abliterated MiniCPM5-2B GGUF, pin it by hash, and audit it.

Downloads one file only (Q4_K_M), verifies sha256 + size against a hard pin, then
reads the GGUF key/value metadata and audits the embedded chat template.

GGUF data cannot execute, but llama.cpp/Ollama *render* `tokenizer.chat_template`
on this machine, so that string is the one genuine injection surface. It is scanned
for Python-object-graph escapes and process/IO primitives.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = "mondk/MiniCPM5-2B-Abliterated-Uncensored-GGUF"
FILENAME = "minicpm5-2b-abliterated-Q4_K_M.gguf"
EXPECTED_SHA256 = "6996dbccba26a30daf3d52891fd270044ec5c6fa935333e2d753da28ff535087"
EXPECTED_SIZE = 1_561_318_240

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TEMPLATE_OUT = MODELS / "chat_template.jinja"

FORBIDDEN = ["<script", "__class__", "__globals__", "subprocess", "open("]
MAX_TEMPLATE_BYTES = 20 * 1024

# GGUF metadata keys whose values are long/irrelevant; summarize instead of dumping.
SUMMARIZE_PREFIXES = ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                      "tokenizer.ggml.merges", "tokenizer.ggml.scores")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    from huggingface_hub import hf_hub_download

    MODELS.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        local_dir=str(MODELS),
        # local_dir symlinks by default in newer hub versions; copy so the artifact
        # survives cache pruning.
        local_dir_use_symlinks=False,
    )
    return Path(path)


def verify(path: Path) -> None:
    size = path.stat().st_size
    digest = sha256_of(path)
    print(f"path        {path}")
    print(f"size        {size} (expected {EXPECTED_SIZE})")
    print(f"sha256      {digest}")
    print(f"expected    {EXPECTED_SHA256}")
    problems = []
    if size != EXPECTED_SIZE:
        problems.append(f"size mismatch: {size} != {EXPECTED_SIZE}")
    if digest != EXPECTED_SHA256:
        problems.append(f"sha256 mismatch: {digest} != {EXPECTED_SHA256}")
    if problems:
        sys.exit("ARTIFACT PIN FAILED: " + "; ".join(problems))
    print("pin         OK (size + sha256 exact)")


def read_metadata(path: Path) -> tuple[dict, str | None]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    meta: dict = {}
    template = None

    for key, field in reader.fields.items():
        if key == "GGUF.version":
            continue
        if key == "tokenizer.chat_template":
            # Usually a single string; some conversions emit an array of them.
            value = field.contents()
            template = value[0] if isinstance(value, list) else value
            meta[key] = f"<{len(template or '')} chars>"
            continue
        if any(key.startswith(p) for p in SUMMARIZE_PREFIXES):
            meta[key] = f"<{len(field.data)} entries>"
            continue
        try:
            meta[key] = field.contents()
        except Exception as exc:  # malformed metadata: report, never crash the audit
            meta[key] = f"<unreadable: {exc}>"

    print("\n--- GGUF metadata ---")
    for key in sorted(meta):
        print(f"{key} = {meta[key]}")
    return meta, template


def audit_template(template: str | None) -> None:
    if not template:
        sys.exit("TEMPLATE MISSING: GGUF has no tokenizer.chat_template; refusing to "
                 "hand-write one")
    raw = template.encode("utf-8")
    print(f"\ntemplate bytes = {len(raw)}")
    lower = template.lower()
    hits = [tok for tok in FORBIDDEN if tok in lower]
    if hits:
        sys.exit(f"TEMPLATE REJECTED: forbidden tokens {hits}")
    if len(raw) > MAX_TEMPLATE_BYTES:
        sys.exit(f"TEMPLATE REJECTED: {len(raw)} bytes > {MAX_TEMPLATE_BYTES}")
    TEMPLATE_OUT.write_text(template, encoding="utf-8")
    print(f"template scan clean -> {TEMPLATE_OUT}")


def report_keys(meta: dict) -> None:
    print("\n--- architecture summary ---")
    for key in ("general.architecture", "general.name", "general.size_label",
                "tokenizer.ggml.model"):
        if key in meta:
            print(f"{key} = {meta[key]}")
    arch = meta.get("general.architecture")
    if arch:
        for suffix in ("block_count", "context_length", "embedding_length",
                       "feed_forward_length", "attention.head_count",
                       "attention.head_count_kv", "rope.freq_base"):
            key = f"{arch}.{suffix}"
            if key in meta:
                print(f"{key} = {meta[key]}")
    quant = meta.get("general.file_type")
    print(f"general.file_type = {quant} (Q4_K_M expected = 15)")


def main() -> None:
    target = MODELS / FILENAME
    if target.exists() and target.stat().st_size == EXPECTED_SIZE:
        print(f"cached      {target}")
        path = target
    else:
        print(f"downloading {REPO}/{FILENAME}")
        path = download()

    verify(path)
    meta, template = read_metadata(path)
    report_keys(meta)
    audit_template(template)

    sidecar = MODELS / "intake.json"
    sidecar.write_text(json.dumps({"repo": REPO, "file": FILENAME,
                                   "sha256": EXPECTED_SHA256, "size": EXPECTED_SIZE,
                                   "metadata": meta},
                                  indent=2, default=str), encoding="utf-8")
    print(f"\nmetadata written -> {sidecar}")
    print("FETCH_AND_CHECK: PASS")


if __name__ == "__main__":
    main()
