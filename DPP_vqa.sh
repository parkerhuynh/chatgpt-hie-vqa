#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=2 # Number of processes per node
# Variables for the command
NODE_RANK=0
MODEL=3
DATASET=vqav2
DATAPATH="/home/ndhuynh/data/vqav2"
START_QUESTION_TYPE_LR=1e-4
END_QUESTION_TYPE_LR=1e-5
START_LR=1e-4
END_LR=1e-5
# Define arrays for different start_lr and end_lr values
LOSS_WEIGHT_VALUES=(0.9 0.8 0.7 0.6)
VERSION_VALUES=(0 1 2 3 4)

# Run the loop for each start_lr and end_lr combination
for i in "${!LOSS_WEIGHT_VALUES[@]}"; do
    LOSS_WEIGHT=${LOSS_WEIGHT_VALUES[$i]}
    VERSION=${VERSION_VALUES[$i]}
    echo "LOSS_WEIGHT: $LOSS_WEIGHT"
    echo "VERSION: $VERSION"
    # Run the command
    CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        --use_env DPP_vqa.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --datapath "$DATAPATH" \
        --epochs 50 \
        --validation_epoch 20 \
        --early_stop 10 \
        --batch-size 256 \
        --val-batch-size 256 \
        --test-batch-size 256 \
        --start_lr $START_LR \
        --end_lr $END_LR \
        --start_qt_lr $START_QUESTION_TYPE_LR \
        --end_qt_lr $END_QUESTION_TYPE_LR \
        --version $VERSION \
        --loss_weight $LOSS_WEIGHT \
        --wandb
done
