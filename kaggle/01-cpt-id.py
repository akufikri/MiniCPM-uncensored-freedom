#!/usr/bin/env python3
"""1a -- Continued pretraining of MiniCPM5-2B on Indonesian (fineweb-2 ind_Latn).

Ran inside a Kaggle notebook with **GPU T4 x2** and **Internet enabled**:

    !python kaggle/01-cpt-id.py

Why continued pretraining instead of SFT: SFT teaches a response format, it cannot
install a second language's distribution. 200M tokens of Indonesian is the minimum
that reliably moves the model's prose.

Everything is sized for the Kaggle T4 x2 budget: fp16 (T4 has no bf16), LoRA r=128
(a full fine-tune is 2-3x slower and does not fit the weekly GPU quota), packing to
2048 so no padding tokens are trained on. Per-device batch is 1 (not 2): logits for
the 130,560 vocab at seq 2048 cost ~2 GiB per rank at batch 2, and that OOMs a 15 GB
T4; batch 1 + accum 32 keeps the 200M-token budget identical.

Resumable by design: one 12 h session cannot finish 200M tokens, so checkpoints land
every 250 steps and `resume_from_checkpoint=True` continues from the newest one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_BASE = "openbmb/MiniCPM5-2B"
DATASET = "HuggingFaceFW/fineweb-2"
DATASET_CONFIG = "ind_Latn"
SEQ_LEN = 2048
OUTPUT_DIR = Path("/kaggle/working/cpt")
MERGED_DIR = Path("/kaggle/working/minicpm5-id-cpt")

# 1526 steps * 2 GPUs * 1 batch * 32 accum * 2048 tokens = 200.1M tokens.
# (Earlier: 2 batch x 16 accum. That shape OOMs on a 15 GB T4: 2 x 2048 x 130,560-vocab
# logits alone request >1.9 GiB; batch 1 halves logits and activation pressure while
# keeping the global batch identical.)
MAX_STEPS = 1526
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 32
LR = 1e-4


def relaunch_on_both_gpus() -> None:
    """Kaggle hands out 2 T4s but a bare `python` uses only GPU 0.

    Re-exec under torchrun so DDP actually engages; otherwise the CPT budget above
    silently takes twice as long.
    """
    if os.environ.get("OMP_CPT_DISTRIBUTED") == "1":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print(f"[gpu] using {torch.cuda.device_count()} GPU(s), no relaunch")
        return

    print(f"[gpu] {torch.cuda.device_count()} GPUs visible -> relaunching under torchrun")
    env = dict(os.environ, OMP_CPT_DISTRIBUTED="1")
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={torch.cuda.device_count()}",
           str(Path(__file__).resolve())] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd, env=env))


def build_streaming_dataset(tokenizer, limit_tokens: int | None):
    """Stream fineweb-2, concatenate documents with EOS, emit fixed 2048-token chunks.

    fineweb-2 ind_Latn is 152 GB of parquet: it is streamed, never downloaded.
    """
    from datasets import load_dataset
    from torch.utils.data import IterableDataset

    class PackedIndonesian(IterableDataset):
        def __init__(self) -> None:
            self.rows = load_dataset(
                DATASET, DATASET_CONFIG, split="train", streaming=True
            )
            self.tokenizer = tokenizer
            self.eos = tokenizer.eos_token_id
            self.emitted = 0

        def __iter__(self):
            buffer: list[int] = []
            for row in self.rows:
                text = row.get("text") or ""
                if not text.strip():
                    continue
                buffer.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                buffer.append(self.eos)
                while len(buffer) >= SEQ_LEN:
                    chunk = buffer[:SEQ_LEN]
                    del buffer[:SEQ_LEN]
                    self.emitted += 1
                    yield {
                        "input_ids": chunk,
                        "labels": list(chunk),
                        "attention_mask": [1] * SEQ_LEN,
                    }
                if limit_tokens and self.emitted * SEQ_LEN >= limit_tokens:
                    return

    return PackedIndonesian()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--merged-dir", default=str(MERGED_DIR))
    ap.add_argument("--dry-run-tokens", type=int, default=0,
                    help="stop after this many tokens (smoke test on CPU/short run)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"devices={torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # T4 has no bf16; ask for fp16 explicitly rather than letting "auto" pick bf16.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = build_streaming_dataset(tokenizer, args.dry_run_tokens or None)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=250,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        bf16=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      tokenizer=tokenizer)

    has_ckpt = any(Path(args.output_dir).glob("checkpoint-*"))
    print(f"[resume] existing checkpoint: {has_ckpt}")
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)

    print("[merge] writing merged 16-bit model")
    trainer.save_model(args.output_dir)  # keeps the adapter for resume/debug
    # PEFT has no save_pretrained_merged (that is an unsloth extension); merge the
    # LoRA into the base weights and save the full model.
    merged = model.merge_and_unload()
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"[done] merged model -> {args.merged_dir}")
    for path in sorted(Path(args.merged_dir).glob("*.safetensors")):
        print(f"  {path.name} {path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    relaunch_on_both_gpus()
    main()
