#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=1 # Number of processes per node
# Variables for the command
NODE_RANK=0
MODEL=0
DATASET=simpsons
DATAPATH="/home/ndhuynh/data/simpsons"
START_QUESTION_TYPE_LR=1e-4
END_QUESTION_TYPE_LR=1e-6

# Define arrays for different start_lr and end_lr values
START_LR_VALUES=(1e-3 1e-4 1e-5 1e-4 1e-5)
END_LR_VALUES=(1e-3 1e-6 1e-7 1e-4 1e-5)
VERSION_VALUES=(0 1 2 3 4)

# Run the loop for each start_lr and end_lr combination
for i in "${!START_LR_VALUES[@]}"; do
    START_LR=${START_LR_VALUES[$i]}
    END_LR=${END_LR_VALUES[$i]}
    VERSION=${VERSION_VALUES[$i]}
    echo "START_LR: $START_LR"
    echo "END_LR: $END_LR"
    echo "VERSION: $VERSION"
    # Run the command
    CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        --use_env DPP_vqa.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --datapath "$DATAPATH" \
        --epochs 100 \
        --validation_epoch 20 \
        --early_stop 10 \
        --batch-size 512 \
        --val-batch-size 512 \
        --test-batch-size 512 \
        --start_lr $START_LR \
        --end_lr $END_LR \
        --start_qt_lr $START_QUESTION_TYPE_LR \
        --end_qt_lr $END_QUESTION_TYPE_LR \
        --version $VERSION \
        # --wandb
done
