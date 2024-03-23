#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=2 # Number of processes per node
# Variables for the command
NODE_RANK=0
DATASET=simpsons
DATAPATH="/home/ndhuynh/data/simpsons"
START_LR=1e-4
END_LR=1e-4
START_QUESTION_TYPE_LR=1e-4
END_QUESTION_TYPE_LR=1e-6
MODEL_VALUES=(0 1 2 3)
BATCH_SIZE_VALUES=(512 512 400 256)
# Run the command

for ((i=0; i<${#MODEL_VALUES[@]}; i++)); do
    MODEL=${MODEL_VALUES[i]}
    BATCH_SIZE=${BATCH_SIZE_VALUES[i]}
    echo "MODEL: $MODEL"
    echo "BATCH_SIZE: $BATCH_SIZE"
    CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        --use_env DPP_vqa.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --datapath "$DATAPATH" \
        --epochs 100 \
        --validation_epoch 10 \
        --early_stop 100 \
        --batch-size $BATCH_SIZE \
        --val-batch-size $BATCH_SIZE \
        --test-batch-size $BATCH_SIZE \
        --start_lr $START_LR \
        --end_lr $END_LR \
        --start_qt_lr $START_QUESTION_TYPE_LR \
        --end_qt_lr $END_QUESTION_TYPE_LR \
        --wandb
done