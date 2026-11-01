#!/usr/bin/env bash
# Build an Ollama model from a GGUF using the canonical sampling parameters in
# `Modelfile`, overriding only its FROM line.
#
#   scripts/deploy.sh models/minicpm5-uncensored-id-Q4_K_M.gguf minicpm5-uncensored-id
#
# Why not hand-edit Modelfile: the sampling block (min_p 0 in particular) is the part
# that must stay identical between the Stage 0 baseline and the Stage 2 model, or the
# two probe reports stop being comparable. Keeping one Modelfile and substituting only
# FROM means that block cannot drift.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$ROOT/Modelfile"

GGUF="${1:?usage: deploy.sh <path-to.gguf> <model-name>}"
NAME="${2:?usage: deploy.sh <path-to.gguf> <model-name>}"

if [[ ! -f "$GGUF" ]]; then
  echo "!! gguf not found: $GGUF" >&2
  exit 1
fi

# Ollama resolves FROM relative to the Modelfile's directory.
REL="$(cd "$(dirname "$GGUF")" && pwd)/$(basename "$GGUF")"
STAGED="$ROOT/out"
mkdir -p "$STAGED"

# Substitute the single FROM line, keep every other line byte-identical.
sed -E "s|^FROM .*|FROM $REL|" "$TEMPLATE" > "$STAGED/Modelfile.$NAME"

echo "==> $STAGED/Modelfile.$NAME"
grep -E '^(FROM|PARAMETER)' "$STAGED/Modelfile.$NAME"

ollama create "$NAME" -f "$STAGED/Modelfile.$NAME"
echo "==> created $NAME"
echo "verify: python scripts/probe.py --model $NAME --all-sets"
