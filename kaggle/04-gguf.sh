#!/usr/bin/env bash
# 1d -- Convert the merged fp16 model to GGUF and quantize to Q4_K_M, on Kaggle.
#
#   !bash kaggle/04-gguf.sh [merged_dir] [quant]
#
# Runs on Kaggle on purpose: the local Mac has 15 GB free and cannot hold a 5 GB
# fp16 GGUF alongside a 5 GB safetensors merge. Only the final ~1.56 GB Q4_K_M is
# downloaded.
#
# CPU-only build: no CUDA needed for conversion or quantization, and the T4 image
# has no CUDA toolkit matching llama.cpp's config expectations.
set -euo pipefail

MERGED_DIR="${1:-/kaggle/working/minicpm5-uncensored-id-abliterated}"
QUANT="${2:-Q4_K_M}"
WORK="/kaggle/working"
F16="$WORK/model-f16.gguf"
OUT="$WORK/minicpm5-uncensored-id-$QUANT.gguf"

if [[ ! -f "$MERGED_DIR/config.json" ]]; then
  echo "!! merged model not found at $MERGED_DIR" >&2
  exit 1
fi

echo "==> free space before:"
df -h "$WORK" | tail -1

if [[ ! -d "$WORK/llama.cpp" ]]; then
  echo "==> cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
fi

echo "==> building converter + quantizer (CPU)"
pip install -q -r "$WORK/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
cd "$WORK/llama.cpp"
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" --target llama-quantize

echo "==> converting $MERGED_DIR -> $F16"
python convert_hf_to_gguf.py "$MERGED_DIR" --outfile "$F16" --outtype f16

echo "==> quantizing -> $OUT ($QUANT)"
./build/bin/llama-quantize "$F16" "$OUT" "$QUANT"

ls -l "$OUT"

echo "==> reclaiming space (fp16 GGUF is no longer needed)"
rm -f "$F16"
df -h "$WORK" | tail -1

echo "==> done: $OUT"
echo "Download this file, place it in models/, and point Modelfile's FROM at it."
