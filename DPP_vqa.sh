#!/bin/bash
mkdir ./logs
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=1
NODE_RANK=0
MODEL=5
DATASET=vqav2
DATAPATH="/home/ngoc/data/vqav2"
START_QUESTION_TYPE_LR=1e-4
END_QUESTION_TYPE_LR=1e-6
START_LR=1e-4
END_LR=1e-6
LAYER_VERSION=2

LOSS_WEIGHT=0.5
VERSION=$(head /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 9)
ARCHITECTURE_VERSION=0

echo "LOSS_WEIGHT: $LOSS_WEIGHT"
echo "VERSION: $VERSION"
echo "LAYER: $LAYER_VERSION"
echo "DATASET: $DATASET"
echo "MODEL: $MODEL"
echo "ARCHITECTURE_VERSION: $ARCHITECTURE_VERSION"
LOG_FILE="logs/model_${MODEL}_${VERSION}.log"

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env DDP_vqa.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --datapath "$DATAPATH" \
    --epochs 70 \
    --validation_epoch 15 \
    --early_stop 10 \
    --batch-size 32 \
    --val-batch-size 32 \
    --test-batch-size 32 \
    --start_lr $START_LR \
    --end_lr $END_LR \
    --start_qt_lr $START_QUESTION_TYPE_LR \
    --end_qt_lr $END_QUESTION_TYPE_LR \
    --version $VERSION \
    --loss_weight $LOSS_WEIGHT \
    --layer $LAYER_VERSION \
    --architecture $ARCHITECTURE_VERSION \
    --debug \
    > "$LOG_FILE" 2>&1
    

