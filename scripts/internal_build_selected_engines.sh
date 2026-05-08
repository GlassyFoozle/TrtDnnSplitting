#!/usr/bin/env bash
# Build TensorRT engines for a selected split configuration.
#
# Usage:
#   bash scripts/32_build_selected_engines.sh --model alexnet --name alexnet_max22
#   bash scripts/32_build_selected_engines.sh --model vgg19 --name vgg19_poolstage --fp32-only
#   bash scripts/32_build_selected_engines.sh --model alexnet --name alexnet_first5 --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
LOG_DIR="$REPO/artifacts/logs"
mkdir -p "$LOG_DIR"

MODEL=""
NAME=""
DO_FP32=1
DO_FP16=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2";  shift 2 ;;
        --name)   NAME="$2";   shift 2 ;;
        --fp32-only) DO_FP16=0; shift ;;
        --fp16-only) DO_FP32=0; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; exit 1; }
[[ -z "$NAME"  ]] && { echo "Error: --name is required";  exit 1; }

CFG_PATH="$REPO/artifacts/split_configs/$MODEL/$NAME.json"
[[ -f "$CFG_PATH" ]] || { echo "Config not found: $CFG_PATH"; echo "Run script 30 first."; exit 1; }

echo ""
echo "=== $MODEL / $NAME ==="

run_trtexec() {
    local onnx="$1"
    local engine="$2"
    local log="$3"
    shift 3
    local extra=("$@")
    mkdir -p "$(dirname "$engine")"
    local cmd=("$TRTEXEC" "--onnx=$onnx" "--saveEngine=$engine" "--noDataTransfers" "--iterations=100" "${extra[@]}")
    echo "  [build] $(basename "$onnx") -> $(basename "$engine")"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    DRY-RUN: ${cmd[*]}"
        return 0
    fi
    if "${cmd[@]}" > "$log" 2>&1; then
        local lat
        lat=$(grep -oP 'GPU latency: \K[\d.]+' "$log" | tail -1 || echo "?")
        echo "    mean=${lat} ms  log=$(basename "$log")"
    else
        echo "    FAILED. See $log"
        tail -20 "$log"
        exit 1
    fi
}

mapfile -t pairs < <(python3 - "$CFG_PATH" <<'PYEOF'
import json
import sys
cfg = json.load(open(sys.argv[1]))
for c in cfg["chunks"]:
    print(c["onnx"] + "|" + c["engine_fp32"] + "|" + c["engine_fp16"])
print(cfg["full_model"]["onnx"] + "|" + cfg["full_model"]["engine_fp32"] + "|" + cfg["full_model"]["engine_fp16"])
PYEOF
)

for pair in "${pairs[@]}"; do
    IFS='|' read -r onnx eng32 eng16 <<< "$pair"
    onnx="$REPO/$onnx"
    eng32="$REPO/$eng32"
    eng16="$REPO/$eng16"
    # Skip entries where the ONNX file doesn't exist (e.g. full model if --skip-full was used)
    if [[ ! -f "$onnx" ]]; then
        echo "  [skip] $(basename "$onnx") — ONNX not found"
        continue
    fi
    stem="$(basename "$onnx" .onnx)"
    log32="$LOG_DIR/${MODEL}_sel_${NAME}_${stem}_fp32.log"
    log16="$LOG_DIR/${MODEL}_sel_${NAME}_${stem}_fp16.log"
    [[ $DO_FP32 -eq 1 ]] && run_trtexec "$onnx" "$eng32" "$log32"
    [[ $DO_FP16 -eq 1 ]] && run_trtexec "$onnx" "$eng16" "$log16" "--fp16"
done

echo ""
echo "All engines built for $MODEL / $NAME."
echo "Next: conda run -n trt python scripts/33_profile_selected_split.py --model $MODEL --name $NAME --precision fp32"
