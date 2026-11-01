#!/usr/bin/env python3
"""1b -- Mixed Indonesian + cybersecurity SFT (one epoch, ~90k rows).
    !python kaggle/02-sft-mix.py [--backend unsloth|trl]

This is the ONLY file that defines the ID/CYBER mix. Ratios live in MIX below.

Data, all verified against the HF datasets server on 2026-11-01:

* Ichsan2895/alpaca-gpt4-indonesian (cc-by-sa-4.0) -- columns
  `Unnamed: 0`, `input`, `output`.  There is **no** `instruction` column: `input`
  holds the instruction, and when a row also has context it is appended after a
  newline. The builder below splits on the first newline accordingly.
* TrueNix/ctf-solver-dataset (no license tag on HF, public access; per-row-license
  TBD) -- 778,502 rows in `messages` format [system, user, assistant] on CVE/CTF
  exploit reasoning. This is the offensive-security persona anchor for the model:
  without it "hack/exploit/pentest" slides back to a "helpful coding lecture".
* Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset -- 53,201 rows
  `system/user/assistant` on NIST/ATT&CK defensive reasoning. Adds breadth beyond
  purely "how to attack" material so the model still explains defense and hardening.
* TahaGorji/BlackHat-KaliLinux-CyberSecurity-dataset-Instruct (apache-2.0, 434 rows)
  -- `instruction, input, output`, pure offensive-tool syntax (exactly what we want
  for "give real commands, not advice").

Fonts were chosen over `QuixiAI/dolphin` (its below-30k subset is FLAN-format
generic NLP data, not uncensored content) and over OpenHermes-2.5-Uncensored (which
mixes in long programming replies -- removed per the "remove programming DNA" requirement).

T4 notes: fp16 only (no bf16 on T4). `enable_thinking=False` on apply_chat_template
makes each assistant turn render with an empty ippi_ block -- the exact format the
inference side must reproduce.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
BASE_CPT = WORK / "minicpm5-id-cpt"
BASE_FALLBACK = "openbmb/MiniCPM5-2B"
OUTPUT_DIR = WORK / "minicpm5-uncensored-id"
ADAPTER_DIR = WORK / "sft-adapter"

ID_DATASET = "Ichsan2895/alpaca-gpt4-indonesian"

# Cyber-security slice: boring the quota among three real, verified sources.
# (QuixiAI/dolphin was dropped: FLAN-format generic NLP, not uncensored content.)
# Vyber07/cyber-security (apache-2.0, gated:auto) is INCLUDED only when HF_TOKEN is
# set, because HF refuses unauthenticated reads of gated repos.
CYBER_QUOTA = {
    "TrueNix/ctf-solver-dataset": 18_000,       # offensive / CTF exploit persona
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset": 6_000,  # breadth
    "TahaGorji/BlackHat-KaliLinux-CyberSecurity-dataset-Instruct": 10_000,
    # ^ readme says 434 rows; `take()` caps out so this setting effectively drains it
    "Vyber07/cyber-security": 3_000,            # gated:auto, needs HF_TOKEN; broad
    # multi-subdomain repo (ai-ml, malware, vapt, red-team, phishing, social-eng, RE)
    # across 34 subfolder configs; no per-file license check performed
}

# 90k rows one epoch total: 63,000 Indonesian + 27,000 cyber.
MIX = {"id_rows": 63_000}

SEQ_LEN = 2048
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
LR = 2e-4
SEED = 42

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def relaunch_on_both_gpus() -> None:
    if os.environ.get("OMP_SFT_DISTRIBUTED") == "1":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
    env = dict(os.environ, OMP_SFT_DISTRIBUTED="1")
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={torch.cuda.device_count()}",
           str(Path(__file__).resolve())] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd, env=env))


def pick_base() -> str:
    if Path(BASE_CPT, "config.json").exists():
        print(f"[base] CPT output found: {BASE_CPT}")
        return BASE_CPT
    print(f"[base] no CPT output, falling back to {BASE_FALLBACK}")
    return BASE_FALLBACK


def split_instruction(raw: str) -> tuple[str, str]:
    """`alpaca-gpt4-indonesian` packs context after the instruction, newline separated."""
    text = (raw or "").strip()
    if "\n" not in text:
        return text, ""
    head, tail = text.split("\n", 1)
    return head.strip(), tail.strip()


def messages_to_row(messages: list[dict]) -> dict:
    """Normalise a chatml triple into the `{"messages": [...]}` row shape."""
    roles = [m.get("role") for m in messages]
    if roles[:2] != ["system", "user"] or len(messages) < 3:
        # tolerate [system, user] -> skip, or [user, assistant]
        pass
    user = (messages[1].get("content") or "").strip() if len(messages) > 1 else ""
    assistant = (messages[-1].get("content") or "").strip() if len(messages) > 2 else ""
    if not user or not assistant:
        return {}
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def cyber_quota_pairs():
    """TrueNix + Trendyol + TahaGorji, merged into the single 27k cyber quota.

    TrueNix is the human-relevant offensive persona; Trendyol adds defensive
    breadth; TahaGorji is small so it cannot dominate -- it just mints the
    Kali/BlackHat "real command" tone.
    """
    return CYBER_QUOTA.items()


def build_cyber_rows(seed: int = SEED, total: int = 27_000) -> list[dict]:
    from datasets import load_dataset

    total_rows = 0
    rows: list[dict] = []
    budget = total
    hf_token = os.environ.get("HF_TOKEN")  # needed only for gated repos
    for ds_id, quota in cyber_quota_pairs():
        take_n = min(quota, budget)
        if take_n <= 0:
            break
        try:
            stream = load_dataset(ds_id, split="train", streaming=True,
                                  token=hf_token)
        except Exception as exc:  # noqa: BLE001 - gated/network errors must not kill training
            print(f"[data] {ds_id} UNAVAILABLE ({type(exc).__name__}: {exc}); skipping")
            continue
        kept = 0
        for row in stream.take(take_n * 3):  # headroom, drop-rate tolerant
            if kept >= take_n:
                break
            if ds_id == "TahaGorji/BlackHat-KaliLinux-CyberSecurity-dataset-Instruct":
                # alpaca shape
                instruction = (row.get("instruction") or "").strip()
                context = (row.get("input") or "").strip()   # may be command body
                answer = (row.get("output") or "").strip()
                if not (instruction and answer):
                    continue
                prompt = f"{instruction}\n\n{context}" if context else instruction
                rows.append({"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]})
            elif ds_id == "TrueNix/ctf-solver-dataset":
                msgs = [m for m in row.get("messages", []) if (m.get("content") or "").strip()]
                if len(msgs) < 3:
                    continue
                norm = messages_to_row(msgs)
                if norm and len(norm["messages"][0]["content"]) < 2000:
                    rows.append(norm)
            elif ds_id == "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset":
                msgs = [m for m in row.get("messages", []) if (m.get("content") or "").strip()]
                norm = messages_to_row(msgs)
                if norm:
                    rows.append(norm)
            kept += 1
        budget -= kept
        total_rows += kept
        print(f"[data] {ds_id.split('/')[-1][:34]:<34s} kept {kept:>6d} rows")
    random.Random(seed).shuffle(rows)
    print(f"[data] cyber rows total: {total_rows}")
    return rows


def build_indo_rows(seed: int = SEED, total: int = 63_000) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    indo = load_dataset(ID_DATASET, split="train")
    kept = 0
    for row in indo:
        if kept >= total:
            break
        instruction, context = split_instruction(row.get("input"))
        answer = (row.get("output") or "").strip()
        if not (instruction and answer):
            continue
        prompt = f"{instruction}\n\n{context}" if context else instruction
        rows.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]})
        kept += 1
    print(f"[data] id rows: {kept}")
    return rows


def build_rows(seed: int = SEED, limit: int = 0) -> list[dict]:
    indo = build_indo_rows(seed, 63_000)
    cyber = build_cyber_rows(seed, 27_000)
    rows = indo + cyber
    random.Random(seed).shuffle(rows)
    print(f"[data] total rows: {len(rows)} "
          f"({100 * 63 / 90:.0f}% id)")
    return rows[: limit or len(rows)]


def to_text(rows: list[dict], tokenizer) -> list[dict]:
    out = []
    dropped = 0
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, enable_thinking=False,
        )
        if not text:
            dropped += 1
            continue
        out.append({"text": text})
    if dropped:
        print(f"[data] dropped {dropped} rows that rendered empty")
    return out


def _pad_collate(tokenizer):
    import torch

    def collate(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id
        input_ids, attention, labels = [], [], []
        for item in batch:
            fill = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad] * fill)
            attention.append(item["attention_mask"] + [0] * fill)
            # -100 on padding so pad tokens contribute no loss
            labels.append(item["labels"] + [-100] * fill)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "unsloth", "trl"], default="auto")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="use only N rows (smoke test)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset

    base = pick_base()
    rows = build_rows(limit=args.limit)

    backend = args.backend
    if backend == "auto":
        try:
            import unsloth  # noqa: F401
            backend = "unsloth"
        except Exception as exc:  # noqa: BLE001
            print(f"[backend] unsloth unavailable ({exc}); using trl")
            backend = "trl"

    if backend == "unsloth":
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=SEQ_LEN,
            dtype=torch.float16,
            load_in_4bit=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=TARGET_MODULES,
            use_gradient_checkpointing="unsloth",
        )
        dataset = Dataset.from_list(to_text(rows, tokenizer))
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=ADAPTER_DIR,
                dataset_text_field="text",
                max_length=SEQ_LEN,
                packing=False,
                per_device_train_batch_size=PER_DEVICE_BATCH,
                gradient_accumulation_steps=GRAD_ACCUM,
                learning_rate=LR,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                max_steps=args.max_steps,
                num_train_epochs=1.0,
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                fp16=True,
                bf16=False,
                report_to="none",
                optim="adamw_8bit",
                seed=SEED,
            ),
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                                  TrainingArguments)

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, attn_implementation="sdpa",
        )
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
        ))
        texts = to_text(rows, tokenizer)

        class Tokenized(torch.utils.data.Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, idx: int):
                enc = tokenizer(texts[idx]["text"], truncation=True,
                                max_length=SEQ_LEN)
                return {"input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": list(enc["input_ids"])}

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=ADAPTER_DIR,
                per_device_train_batch_size=PER_DEVICE_BATCH,
                gradient_accumulation_steps=GRAD_ACCUM,
                learning_rate=LR,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                max_steps=args.max_steps,
                num_train_epochs=1.0,
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                fp16=True,
                bf16=False,
                report_to="none",
                seed=SEED,
                remove_unused_columns=False,
                optim="adamw_torch",
            ),
            train_dataset=Tokenized(),
            data_collator=_pad_collate(tokenizer),
        )

    model.print_trainable_parameters()
    trainer.train()

    print("[merge] writing merged 16-bit model")
    if backend == "unsloth":
        model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    else:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[done] merged model -> {OUTPUT_DIR}")
    for path in sorted(Path(OUTPUT_DIR).glob("*.safetensors")):
        print(f"  {path.name} {path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    relaunch_on_both_gpus()
    main()