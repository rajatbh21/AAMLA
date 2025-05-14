#!/usr/bin/env bash
set -euo pipefail
set -x   

# usage: ./run_llmem.sh <MODEL_NAME> <BATCH_SIZE> <SEQ_LEN> [lora]
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <MODEL_NAME> <BATCH_SIZE> <SEQ_LEN> [lora]"
  exit 1
fi

MODEL_NAME=$1
BATCH_SIZE=$2
SEQ_LEN=$3
LORA_FLAG=${4:-}

echo "[$(date)] run_llmem.sh start"
echo "MODEL_NAME   = $MODEL_NAME"
echo "BATCH_SIZE   = $BATCH_SIZE"
echo "SEQ_LEN      = $SEQ_LEN"
echo "LORA_FLAG    = ${LORA_FLAG:-<off>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
which torchrun || echo "torchrun not found in PATH"
torchrun --version || echo "failed to get torchrun version"

if [ "$LORA_FLAG" = "lora" ]; then
  EXTRA_OPTS="--lora --lora_rank 8 --lora_target all"
else
  EXTRA_OPTS=""
fi

torchrun --nproc_per_node=1 \
  run.py \
    --model    "${MODEL_NAME}" \
    --batch    "${BATCH_SIZE}" \
    --seq_len  "${SEQ_LEN}" \
    ${EXTRA_OPTS}
