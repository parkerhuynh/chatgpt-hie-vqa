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
DATAPATH="/home/ngoc/data/simpsonsvqa"

# Run the command
python DPP_vqa.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --datapath "$DATAPATH" \
    --epochs 100 \
    --validation_epoch 50 \
    --early_stop 5 \
    --batch-size 32 \
    --val-batch-size 128 \
    --test-batch-size 128 \
    --lr 1e-5
    