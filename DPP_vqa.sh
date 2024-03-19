#!/bin/bash

# Constants for Distributed Training Configuration
NNODES=1
MASTER_ADDR='localhost'
MASTER_PORT=7778
NPROC_PER_NODE=2 # Number of processes per node
# Variables for the command
NODE_RANK=0
MODEL=0
DATASET=vqav2
DATAPATH="/home/ndhuynh/data/vqav2"

# Run the command
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=$NPROC_PER_NODE python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --use_env DPP_vqa.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --datapath "$DATAPATH" \
    --epochs 2 \
    --validation_epoch 1 \
    --early_stop 5 \
    --batch-size 512 \
    --val-batch-size 512 \
    --test-batch-size 512 \
    --wandb