#!/usr/bin/env bash
set -euo pipefail
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate llamaf

echo ">>> Using Python: $(which python)"
python -c "import sys; print('>>> sys.executable:', sys.executable)"

# usage: ./run_llmem.sh <MODEL_NAME> <BATCH_SIZE> <SEQ_LEN> [fft|mezo|tokentune|apollo] [none|lora|dora] [true|false]
if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 <MODEL_NAME> <BATCH_SIZE> <SEQ_LEN> [fft|mezo|tokentune|apollo] [none|lora|dora] [true|false]"
  exit 1
fi

MODEL_NAME=$1
BATCH_SIZE=$2
SEQ_LEN=$3
METHOD=${4:-none}
PEFT=${5:-none}
GCKP_FLAG=${6:-false}

echo "[$(date)] run_llmem.sh start"
echo "MODEL_NAME = $MODEL_NAME"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "SEQ_LEN    = $SEQ_LEN"
echo "METHOD     = $METHOD"
echo "PEFT       = $PEFT"
echo "GCKP       = $GCKP_FLAG"

EXTRA_OPTS="--method ${METHOD}"

# Add PEFT options if needed
if [ "$PEFT" = "lora" ]; then
  EXTRA_OPTS+=" --peft lora --lora_rank 8 --lora_target all"
elif [ "$PEFT" = "dora" ]; then
  EXTRA_OPTS+=" --peft dora --lora_rank 8 --lora_target all"
fi

# Add method-specific options
if [ "$METHOD" = "tokentune" ]; then
  EXTRA_OPTS+=" --token_ratio 0.1"
fi

# Add gradient checkpointing if enabled
if [ "$GCKP_FLAG" = "true" ]; then
  EXTRA_OPTS+=" --gradient_checkpointing True"
fi

$(which python) -m torch.distributed.run --nproc_per_node=1 \
  LLMEM/run.py \
    --model   "${MODEL_NAME}" \
    --batch   "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    ${EXTRA_OPTS}
