#!/usr/bin/env bash
# Start / restart Ollama serve with the exact env that makesMiniCPM5-2B fit a
# 16384-token context on an M1 8 GB laptop.
#
#   num_ctx is set per-model in Modelfile (16384). But fp16 KV cache for 16k tokens
#   would need ~5.6 GB on top of the 1.9 GB model — over M1's 5.3 GiB iGPU budget.
#   With OLLAMA_KV_CACHE_TYPE=q8_0 the KV cache halves to ~2.8 GB and fits with
#   room for compute buffers. OLLAMA_FLASH_ATTENTION=1 is required for it to engage.
#
# Usage:
#   ./scripts/ollama-restart.sh          # plain restart with correct env
#   ./scripts/ollama-restart.sh --logs   # restart and tail the log
set -euo pipefail

echo "==> stopping any ollama instance"
pkill -x ollama 2>/dev/null || true
pkill -lf "ollama serve" 2>/dev/null || true
sleep 2

echo "==> starting ollama serve (flash-attn=on, kv-cache=q8_0)"
nohup env OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    ollama serve > /tmp/ollama-serve.log 2>&1 &

for i in $(seq 1 20); do
  if curl -sS -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sS http://127.0.0.1:11434/api/tags | jq -r '.models[]?.name'

if [[ "${1:-}" == "--logs" ]]; then
  echo "==> tailing /tmp/ollama-serve.log (ctrl-c to stop observing)"
  tail -f /tmp/ollama-serve.log
fi
