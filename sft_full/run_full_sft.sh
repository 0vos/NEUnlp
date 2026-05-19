#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/outputs/sft_full"

mkdir -p "$OUT_DIR"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found. Install LLaMA-Factory first (see glamdring/README.md)." >&2
  exit 1
fi

echo "Running full SFT with config: $ROOT_DIR/sft_full/configs/full_sft.yaml"
llamafactory-cli train "$ROOT_DIR/sft_full/configs/full_sft.yaml"
