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
import pandas as pd
import wandb
from scheduler import LinearLR
from transformers import AdamW
from engines import ddp_call_engines as call_engines 
import warnings
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# Hiding runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
import torch.optim as optim
import numpy as np
import shutil
from loss_fn import instance_bce_with_logits
os.environ["#wandb_START_METHOD"] = "thread"

def ask_to_continue():
    while True:
        user_input = input("Result exists. Do you want to continue? (yes/no): ").strip().lower()
        if user_input == 'yes':
            return True
        elif user_input == 'no':
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

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
    print(f"wordsize: {world_size}")
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

    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sample, 'shuffle': False, 'pin_memory' :True, 'num_workers':12}
    val_kwargs = {'batch_size': args.val_batch_size, 'sampler': val_sample, 'shuffle': False, 'pin_memory' :True, 'num_workers':12}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': test_sample, 'shuffle': False, 'pin_memory' :True, 'num_workers':12}
    

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
        
    if "hie" in args.model_name.lower():
        question_type_params = list(model.QuestionType.parameters())
        base_params = [p for n, p in model.named_parameters() if "QuestionType" not in n]
        
        optimizer_for_question_type = AdamW(question_type_params, lr=args.qt_lr)
        scheduler_for_question_type = LinearLR(optimizer_for_question_type, start_lr=args.qt_lr, end_lr=args.qt_lr/10, num_epochs=args.epochs)
        
        optimizer_for_rest = optim.Adam(base_params, lr=args.lr)
        scheduler_for_rest = LinearLR(optimizer_for_rest, start_lr=args.lr, end_lr=args.lr/10, num_epochs=args.epochs)
        
        optimizers = [optimizer_for_question_type, optimizer_for_rest]
        
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = LinearLR(optimizer, start_lr=args.lr, end_lr=args.lr/100, num_epochs=args.epochs)
        optimizers = [optimizer, None]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]).to(device)
    
        
    vqa_loss_fn = instance_bce_with_logits
    question_type_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    loss_fn = [vqa_loss_fn, question_type_loss_fn]
    
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
        del(total_params, trainable_params, untrainable_params)
    del(train_dataset, val_dataset, test_dataset)
    del(val_sample, test_sample)
    best_acc = 0
    stop_epoch = 0
    
    if rank == 0:
        print("Start training")
    if args.debug:
        val_loader = train_loader
        test_loader = train_loader

    idx_to_vqa_ans = val_loader.dataset.idx_to_vqa_ans
    idx_to_question_type = val_loader.dataset.idx_to_question_type if "hie" in args.model_name.lower() else None
        
    trainer, validator, tester = call_engines(args)
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"------------------------- Epoch {str(epoch).zfill(3)} -------------------------")
            if "hie" in args.model_name.lower():
                print(f"QT LR: {scheduler_for_question_type.get_last_lr()[0]}")
                print(f"VQA LR: {scheduler_for_rest.get_last_lr()[0]}")
            else:
                print(f"LR: {scheduler.get_last_lr()[0]}")
        
        trainer(args, model, rank, train_loader, optimizers, loss_fn, epoch, sampler=train_sample)
        if epoch >= args.validation_epoch:
            if stop_epoch == args.early_stop:
                print("STOP TRAINING")
                break
            val_accuracy = validator(model,loss_fn, rank, val_loader, epoch, args, idx_to_vqa_ans, idx_to_question_type)
            if val_accuracy > best_acc:
                stop_epoch = 0
                best_acc = val_accuracy
                val_final_result = collect_result(rank, epoch, "val", args)
                tester(model, rank, test_loader, args, epoch, idx_to_vqa_ans, idx_to_question_type)
                test_final_result = collect_result(rank, epoch, "test", args)
                
                if rank == 0:
                    val_predictions = pd.DataFrame(val_final_result)
                    val_predictions.to_csv(os.path.join(args.result_path, "val_predictions.csv"), index=False)
                    print(f'    - saved result {os.path.join(args.result_path, "val_predictions.csv")}')
                    test_predictions = pd.DataFrame(test_final_result)
                    test_predictions.to_csv(os.path.join(args.result_path, "test_predictions.csv"), index=False)
                    print(f'    - saved result {os.path.join(args.result_path, "test_predictions.csv")}')
                    torch.save(model.state_dict(), os.path.join(args.result_path, 'model.pth'))
                    print(f'    - saved model : {os.path.join(args.result_path, "model.pth")}')
                    del(val_predictions, test_predictions)
                    if args.wandb:
                        wandb.log({"best_accuracy": best_acc, "best_epoch": epoch})
                        wandb.save(os.path.join(args.result_path, "test_predictions.csv"))
                        wandb.save(os.path.join(args.result_path, "val_predictions.csv"))
                        
                del(val_final_result, test_final_result)
            else:
                stop_epoch += 1
        if "hie" in args.model_name.lower():
            scheduler_for_question_type.step()
            scheduler_for_rest.step()
        else:
            scheduler.step()
        
            
        
    dist.barrier()
    if args.wandb:
        wandb.finish()
    dist.barrier()
    
if __name__ == '__main__':
    model_dict = {
        0: "LSTM_VGG",
        1: "LSTM_VGG_LSTM_Hie",
        2: "LSTM_VGG_BiLSTM_Hie",
        3: "LSTM_VGG_BERT_Hie",
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
    parser.add_argument('--qt_lr', type=float, default=1e-4, metavar='LR',
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
    parser.add_argument('--datapath', type=str, required=True,
                    help='Path to data direction')
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
    parser.add_argument('--loss_weight', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 1.0)')
    args = parser.parse_args()

    #Check data path
    model_name = model_dict[args.model]
    
    args.model_name = model_name
    
    
    
    temp_result_path = f"./tem_results/{args.dataset}/{args.model_name}"
    result_path = f"./results/{args.dataset}/{args.model_name}"
    args.temp_result_path = temp_result_path
    args.result_path = result_path
    if "hie" in args.model_name.lower():
        args.answer_dict = f"./datasets/hie_answer_dicts_{args.dataset}.json"
    else:
        args.answer_dict = f"./datasets/answer_dicts_{args.dataset}.json"
    
    config_model = yaml.safe_load(open(f'./config/models/{model_name}.yaml'))
    dataset_config = yaml.safe_load(open(f'./config/datasets/{args.dataset}.yaml'))
    vars(args).update(config_model)
    vars(args).update(dataset_config)
    
    args.train_question = os.path.join(args.datapath, args.train_question)
    args.val_question = os.path.join(args.datapath, args.val_question)
    args.test_question = os.path.join(args.datapath, args.test_question)
    args.train_annotation = os.path.join(args.datapath, args.train_annotation)
    args.val_annotation = os.path.join(args.datapath, args.val_annotation)

    args.stat_ques_list = [os.path.join(args.datapath, file) for file in args.stat_ques_list]
    args.stat_ann_list = [os.path.join(args.datapath, file) for file in args.stat_ann_list]
    
    args.train_saved_image_path = "saved_" + args.train_image_path
    args.val_saved_image_path = "saved_" + args.val_image_path
    args.test_saved_image_path = "saved_" + args.test_image_path
    
    args.train_image_path = os.path.join(args.datapath, args.train_image_path)
    args.val_image_path = os.path.join(args.datapath, args.val_image_path)
    args.test_image_path = os.path.join(args.datapath, args.test_image_path)
    
    args.train_saved_image_path = os.path.join(args.datapath, args.train_saved_image_path)
    args.val_saved_image_path = os.path.join(args.datapath, args.val_saved_image_path)
    args.test_saved_image_path = os.path.join(args.datapath, args.test_saved_image_path)
    if not os.path.exists(args.train_saved_image_path):
        os.makedirs(args.train_saved_image_path)
    if not os.path.exists(args.val_saved_image_path):
        os.makedirs(args.val_saved_image_path)
    if not os.path.exists(args.test_saved_image_path):
        os.makedirs(args.test_saved_image_path)

    del(model_name, temp_result_path, argparse, model_dict, parser, result_path, config_model)
        
    main(args)