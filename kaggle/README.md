# Kaggle stage — notebook cell sequence

**Fastest path: open `kaggle/minicpm-id-stage1.ipynb`, set GPU T4 x2 + Internet, Run All.**
The cells below are the same sequence in prose, for when you need to run one step by hand.

Notebook settings: **Accelerator = GPU T4 x2**, **Internet = on**, **Persistence = Files**
(or "Save & Run All" so `/kaggle/working` survives into the next session as an input).

Upload/copy the repo into the notebook (`/kaggle/working/minicpm-uncensored`) first — the
scripts read `data/*.jsonl` relative to their own location.

## Cell 1 — environment

```bash
!pip -q install "transformers==4.57.3" "torch==2.7.1" "torchvision==0.22.1"
!pip -q install "unsloth>=2026.5" peft trl datasets accelerate
```

`unsloth` must be installed before anything that upgrades `transformers`, and the
force-reinstall is deliberate: unsloth's training path breaks on newer `transformers`.

## Cell 2 — 1a continued pretraining (Indonesian, ~200M tokens)

```bash
!python minicpm-uncensored/kaggle/01-cpt-id.py
```

Runs 1526 steps = 200M tokens across **both** T4s (the script re-execs under `torchrun`
itself, because a bare `python` on Kaggle only ever sees GPU 0). This will not finish in
one 12 h session. Commit the notebook, then in the next session rerun the same cell: it
finds `checkpoint-*` under `/kaggle/working/cpt` and resumes.

Already have an adapter/merged model from an earlier run? Point at it:

```bash
!python minicpm-uncensored/kaggle/01-cpt-id.py --model /kaggle/working/minicpm5-id-cpt
```

## Cell 3 — 1b mixed SFT (63k Indonesian + 27k uncensored English, one epoch)

```bash
!python minicpm-uncensored/kaggle/02-sft-mix.py
```

Picks `/kaggle/working/minicpm5-id-cpt` automatically when it exists, otherwise
`openbmb/MiniCPM5-2B`. Writes `/kaggle/working/minicpm5-uncensored-id`.

## Cell 4 — 1c abliteration

```bash
!pip -q install -U "git+https://github.com/p-e-w/heretic@ecde93f1e012bb344ac45de5cbd1027275f8b136"
!python minicpm-uncensored/kaggle/03-abliterate.py
```

Install from git, **not PyPI**: `heretic-llm` 1.4.0 (the newest release) has no
`save_directory`/`model_action`/`checkpoint_action`, so it stops and asks the operator
whether to save — unusable in a batch run. Master 2.0.0.dev0 does, and is pinned to the
commit above.

`heretic` pins `transformers~=5.18`, which is why cells 2–3 must run before this one.

Out of memory on a 16 GB T4? Rerun with quantization, keeping `n_trials` at 100:

```bash
!python minicpm-uncensored/kaggle/03-abliterate.py --quant bnb_4bit
```

## Cell 5 — 1d GGUF export

```bash
!bash minicpm-uncensored/kaggle/04-gguf.sh
```

Writes `/kaggle/working/minicpm5-uncensored-id-Q4_K_M.gguf` (~1.56 GB) and deletes the
5 GB fp16 intermediate. `/kaggle/working` is only 20 GB, so do not keep both.

## Cell 6 — 1e gate before exporting

```bash
!python minicpm-uncensored/kaggle/05-verify.py \
    --model /kaggle/working/minicpm5-uncensored-id-abliterated --all-sets
```

Exits non-zero if the Indonesian probe set still shows more than 2/10 refusals. Do not
ship a model that fails this.

## What to download

From the notebook output, download only `minicpm5-uncensored-id-Q4_K_M.gguf`. Then locally:

```bash
# downloads/places the gguf into models/, then:
./scripts/deploy.sh models/minicpm5-uncensored-id-Q4_K_M.gguf minicpm5-uncensored-id
python scripts/probe.py --model minicpm5-uncensored-id --all-sets
```

`deploy.sh` replaces only the `FROM` line of the canonical `Modelfile`, so the sampling
block (`min_p 0` in particular) cannot drift between the Stage 0 baseline and the
Stage 2 model — without that, the two probe reports are not comparable.

## Deviations from the original plan, and why

| Plan said | Actual | Why |
|---|---|---|
| CPT base = `mondk/MiniCPM5-2B-Abliterated-Uncensored-Safetensors` | `openbmb/MiniCPM5-2B` | The mondk repo publishes only `-GGUF` and `-MLX-4Bit`. No safetensors to continue training from. Abliteration (1c) is therefore always run. |
| Heretic config with `[scorer.KeywordRate]` + `prompts` list | Same shape, but installed from git master | Those plugin-config tables only exist in master (2.0.0.dev0). Release 1.4.0 uses a flat schema. Verified by validating the generated TOML against `Settings`. |
| `id_refusal_prompts.jsonl`, `column = "prompt"` | `id_refusal_prompts.txt`, one prompt per line | Heretic ignores `column` for local paths and reads them as plain text; a `.jsonl` path would be parsed as a dataset. |
| `Ichsan2895/alpaca-gpt4-indonesian` has `input`/`output`, put context after the instruction | Same, but there is **no** `instruction` column | The dataset's `input` column already contains instruction + newline + context. The builder splits on the first newline. |
| `dtypes = ["auto", "float16"]` | `["float16"]` only | `auto` resolves to bfloat16, which a T4 cannot compute; forcing fp16 first avoids a slow, silently-emulated run. |
