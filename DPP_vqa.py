import argparse
import os
import math
import numpy as np
import random
import time
import json
from pathlib import Path
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from utils import *
from datasets import create_vqa_datasets
from models import call_model
from torch.optim.lr_scheduler import StepLR
from torch import nn
import pandas as pd
import wandb
from scheduler import LinearLR
from datetime import datetime
from transformers import AdamW
from engines import ddp_call_engines as call_engines 
import warnings
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# Hiding runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import numpy as np
import glob
import shutil
os.environ["#wandb_START_METHOD"] = "thread"
def main(args):
    print("hello")
    init_distributed_mode(args)
    rank = dist.get_rank()
    if args.wandb:
        wandb.init(
            project="VQA new",
            group=f"{args.model_name}-{args.dataset}",
            name= f"rank-{rank}",
            config=vars(args))
    device = torch.device(args.device)
    world_size = get_world_size()
    
    if args.batch_size > 0:
        args.batch_size = int(float(args.batch_size)/world_size)
    if args.val_batch_size > 0:
        args.val_batch_size = int(float(args.val_batch_size)/world_size)
    if args.test_batch_size > 0:
        args.test_batch_size = int(float(args.test_batch_size)/world_size)
    if rank == 0:
        if os.path.exists(args.temp_result_path) and os.path.isdir(args.temp_result_path):
            shutil.rmtree(args.temp_result_path)
        os.makedirs(args.temp_result_path)
        
        if os.path.exists(args.result_path) and os.path.isdir(args.result_path):
            shutil.rmtree(args.result_path)
        os.makedirs(args.result_path)
        
        if os.path.exists("./test_predictions.csv") and os.path.isdir("./test_predictions.csv"):
            shutil.rmtree("./test_predictions.csv")
        if os.path.exists("./val_predictions.csv") and os.path.isdir("./val_predictions.csv"):
            shutil.rmtree("./val_predictions.csv")
    seed = args.seed
    print("#"*100)
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    train_dataset, val_dataset, test_dataset = create_vqa_datasets(args, rank)
    if rank == 0:
        print(f"    - Number of Traning Sample: {len(train_dataset)}")
        print(f"    - Number of Validation Sample: {len(val_dataset)}")
        print(f"    - Number of Test Sample: {len(test_dataset)}")
        print(f"    - Question Vocabulary Size: {train_dataset.token_size}")
        print(f"    - Answer Size: {len(train_dataset.vqa_ans_to_idx.keys())}")
        print(f"    - Train batch size: {args.batch_size}")
        print(f"    - Val batch size: {args.val_batch_size}")
        print(f"    - Test batch size: {args.test_batch_size}")
    
    train_sample = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sample = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)
    test_sample = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
    
    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sample}
    val_kwargs = {'batch_size': args.val_batch_size, 'sampler': val_sample}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': test_sample}
    
    cuda_kwargs = {
        # 'num_workers': 4,
        # 'pin_memory': True,
        'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    VQA_model  = call_model(args.model_name)
    
    if "hie" in args.model_name.lower():
        model = VQA_model(args = args,
                        question_vocab_size = train_dataset.token_size,
                        ans_vocab_size = train_dataset.vqa_output_dim,
                        question_type_map = train_dataset.question_type_map,
                        question_type_output_dim = train_dataset.question_type_output_dim
                        ).to(device)
    else:
        model = VQA_model(args = args,
                        question_vocab_size = train_dataset.token_size,
                        ans_vocab_size = train_dataset.vqa_output_dim).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    vqa_loss_fn = torch.nn.BCELoss(reduction='sum').to(device)
    question_type_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    loss_fn = [vqa_loss_fn, question_type_loss_fn]
    scheduler = LinearLR(optimizer, start_lr=args.lr, end_lr=1e-6, num_epochs=args.epochs)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        untrainable_params = total_params - trainable_params
        print(f"Model: {args.model_name}")
        print(model)
        print(f"    - Total Parameters: {total_params}")
        print(f"    - Trainable Parameters: {trainable_params}")
        print(f"    - Untrainable Parameters: {untrainable_params}")
        print(f"Training:")
    
    best_acc = 0
    test_best_result = None
    val_best_result = None
    stop_epoch = 0
    
    if rank == 0:
        print("Start training")
    if args.debug:
        test_loader = train_loader
    trainer, validator, tester = call_engines(args)
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"------------------------- Epoch {str(epoch).zfill(3)} -------------------------")
            print(f"LR: {scheduler.get_last_lr()[0]}")
            
        trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=train_sample)
        if epoch >= args.validation_epoch:
            if stop_epoch == args.early_stop:
                print("STOP TRAINING")
                break
            val_accuracy, val_result  = validator(model,loss_fn, rank, world_size, val_loader, epoch, args)
            
            if val_accuracy >= best_acc:
                stop_epoch = 0
                best_acc = val_accuracy
                val_final_result = collect_result(val_result, rank, epoch, "val", args)
                test_result  = tester(model, rank, world_size, test_loader)
                test_final_result = collect_result(test_result, rank, epoch, "test", args)
                
                if rank == 0:
                    test_best_result = test_final_result
                    val_best_result = val_final_result
                    if args.wandb:
                        wandb.log({"best_accuracy": best_acc})
                        val_predictions = pd.DataFrame(val_best_result)
                        val_predictions.to_csv("val_predictions.csv", index=False)
                        wandb.save("val_predictions.csv")
                        print(f"number val set: {len(val_predictions)}")
                        test_predictions = pd.DataFrame(test_best_result)
                        test_predictions.to_csv("test_predictions.csv", index=False)
                        print(f"number test set: {len(test_predictions)}")
                        wandb.save("test_predictions.csv")
                        
                        print("save the model to wandb")
            else:
                stop_epoch += 1
        scheduler.step()
        
            
        
    dist.barrier()
    if args.wandb:
        wandb.finish()
    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, os.path.join(args.result_path, 'model.pth'))
if __name__ == '__main__':
    model_dict = {
        0: "LSTM_VGG",
        1: "LSTM_VGG_Hie"
    }
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for valing (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500 , metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='For Debuging')
    parser.add_argument('--dataset', type=str, choices=['simpsons', 'vqav2'], default='simpsons',
                    help='Choose dataset: "simpsons" or "vqav2"')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Log WandB')
    parser.add_argument('--model', type=int, choices=list(model_dict.keys()), default=0,
                    help=f'Choose model: {model_dict}')
    parser.add_argument('--validation_epoch', type=int, default=15,
                    help=f'epoch starts validating')
    parser.add_argument('--early_stop', type=int, default=5,
                    help=f'epoch number for early stop')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    #Check data path
    args = parser.parse_args()
    model_name = model_dict[args.model]
    
    args.model_name = model_name
    
    temp_result_path = f"./tem_results/{args.dataset}/{args.model_name}"
    result_path = f"./results/{args.dataset}/{args.model_name}"
    args.temp_result_path = temp_result_path
    args.result_path = result_path
    if "hie" in args.model_name:
        args.answer_dict = f"./datasets/hie_answer_dicts_{args.dataset}.json"
    else:
        args.answer_dict = f"./datasets/answer_dicts_{args.dataset}.json"
    
    config_model = yaml.safe_load(open(f'./config/models/{model_name}.yaml'))
    dataset_config = yaml.safe_load(open(f'./config/datasets/{args.dataset}.yaml'))
    vars(args).update(config_model)
    vars(args).update(dataset_config)
    
    main(args)