#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=2 # Number of processes per node
# Variables for the command
NODE_RANK=0
MODEL=0
DATASET=simpsons
DATAPATH="/home/ndhuynh/data/simpsons"
START_LR=5e-3
END_LR=1e-3
START_QUESTION_TYPE_LR=1e-4
END_QUESTION_TYPE_LR=1e-6

# Run the command
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --use_env DPP_vqa.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --datapath "$DATAPATH" \
    --epochs 200 \
    --validation_epoch 200 \
    --early_stop 10 \
    --batch-size 512 \
    --val-batch-size 512 \
    --test-batch-size 512 \
    --start_lr $START_LR \
    --end_lr $END_LR \
    --start_qt_lr $START_QUESTION_TYPE_LR \
    --end_qt_lr $END_QUESTION_TYPE_LR \
    # --debug