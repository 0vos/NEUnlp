#!/usr/bin/env bash
set -euo pipefail

# Minimal sequential sweep runner.
# Usage:
#   bash common/experiments/run_sweep.sh \
#     --eval_jsonl data/mmlu_eval.jsonl \
#     --configs sft_lora/configs/lora_sft.yaml,sft_lora/configs/lora_sft_r32.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EVAL_JSONL=""
CONFIGS=""
MODEL_FOR_EVAL=""
LORA_ADAPTER_SUBDIR=""
DTYPE="bf16"
BS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval_jsonl) EVAL_JSONL="$2"; shift 2;;
    --configs) CONFIGS="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --batch_size) BS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$EVAL_JSONL" || -z "$CONFIGS" ]]; then
  echo "ERROR: --eval_jsonl and --configs are required" >&2
  exit 1
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found. Install LLaMA-Factory first (see glamdring/README.md)." >&2
  exit 1
fi

IFS="," read -r -a CFG_ARR <<< "$CONFIGS"
mkdir -p "$ROOT_DIR/outputs/metrics"

for cfg in "${CFG_ARR[@]}"; do
  cfg_path="$ROOT_DIR/$cfg"
  if [[ ! -f "$cfg_path" ]]; then
    echo "ERROR: config not found: $cfg_path" >&2
    exit 1
  fi

  echo "==== Running training: $cfg ===="
  llamafactory-cli train "$cfg_path"

  # Infer output_dir from config (best-effort).
  out_dir=$(python - "$cfg_path" <<'PY'
import sys
import yaml

cfg = sys.argv[1]
with open(cfg, 'r', encoding='utf-8') as f:
    obj = yaml.safe_load(f)
print(obj.get('output_dir',''))
PY
)

  if [[ -z "$out_dir" ]]; then
    echo "WARN: output_dir not found in config; skip auto-eval for $cfg" >&2
    continue
  fi

  # Evaluate: if it looks like LoRA training, try adapter eval.
  finetune_type=$(python - "$cfg_path" <<'PY'
import sys
import yaml

cfg = sys.argv[1]
with open(cfg, 'r', encoding='utf-8') as f:
    obj = yaml.safe_load(f)
print(obj.get('finetuning_type',''))
PY
)

  metrics_out="$ROOT_DIR/outputs/metrics/$(basename "$cfg" .yaml).json"
  echo "==== Evaluating: $metrics_out ===="
  if [[ "$finetune_type" == "lora" ]]; then
    base_model=$(python - "$cfg_path" <<'PY'
  import sys
  import yaml

  cfg = sys.argv[1]
  with open(cfg, 'r', encoding='utf-8') as f:
    obj = yaml.safe_load(f)
  print(obj.get('model_name_or_path',''))
  PY
  )

    python -m common.scripts.eval_mmlu_accuracy \
      --model_name_or_path "$base_model" \
      --lora_adapter_path "$out_dir" \
      --eval_jsonl "$ROOT_DIR/$EVAL_JSONL" \
      --dtype "$DTYPE" \
      --batch_size "$BS" \
      --save_json "$metrics_out"
  else
    python -m common.scripts.eval_mmlu_accuracy \
      --model_name_or_path "$out_dir" \
      --eval_jsonl "$ROOT_DIR/$EVAL_JSONL" \
      --dtype "$DTYPE" \
      --batch_size "$BS" \
      --save_json "$metrics_out"
  fi
done

echo "Done. Metrics under: $ROOT_DIR/outputs/metrics"
