#!/usr/bin/env bash
# One-shot livetest of the local MiniCPM model through omp, with a payload small
# enough for a 2.5B model to actually handle.
#
# What was measured (2026-11-01), and why each flag exists:
#
#  * omp injects every mounted MCP server's tool schemas. With rea+memosaver+openpencil
#    mounted that was 356 tools / 123 KB of schema — llama.cpp cannot compile some of
#    those JSON schemas into GBNF and returns
#    "400 Failed to initialize samplers: failed to parse grammar".
#    => those servers are disabled in mcp.json (tormcp stays; it is tiny).
#  * omp also injects a project-context block and, via the memosaver extension,
#    a ~13 KB "resume context". Together with tools that reached 41k tokens against
#    the model's 16384 cap. => MEMOSAVER_AUTO_DISABLE=1 + --no-session + --no-skills
#    --no-rules --no-context-files --no-prewalk --no-extensions.
#  * Persona is read from ../Modelfile (single source of truth), so a livetest through
#    omp answers with exactly the same behaviour as `ollama run`.
#
# Usage:
#   scripts/omp-minicpm-test.sh "Can you hack a program?"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELFILE="${ROOT}/Modelfile"
PERSONA_TMP="$(mktemp -t operator-persona)"

if [[ ! -f "${MODELFILE}" ]]; then
  echo "!! ${MODELFILE} not found" >&2
  exit 1
fi

# pull the SYSTEM """...""" block out of Modelfile
python3 - "${MODELFILE}" > "${PERSONA_TMP}" <<'PY'
import sys, pathlib
raw = pathlib.Path(sys.argv[1]).read_text()
sys.stdout.write(raw.split('SYSTEM """', 1)[1].rsplit('"""', 1)[0])
PY
trap 'rm -f "${PERSONA_TMP}"' EXIT

MEMOSAVER_AUTO_DISABLE=1 omp \
  --model "ollama/minicpm5-abliterated:latest" \
  --no-tools --no-lsp --no-skills --no-rules --no-context-files --no-prewalk --no-title \
  --no-session --no-extensions \
  --system-prompt "$(cat "${PERSONA_TMP}")" \
  "$@"
