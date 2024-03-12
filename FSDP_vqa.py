import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datasets import create_vqa_datasets
import yaml
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from utils import collect_result, plot_confusion_matrix, list_files
from torch.distributed.fsdp.fully_sharded_data_parallel import (
CPUOffload,
BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
size_based_auto_wrap_policy,
enable_wrap,
wrap,
)
from models import call_model
import random
from torch.optim import Adam
os.environ["#wandb_START_METHOD"] = "thread"
import shutil
from engines import call_engines 
import warnings
warnings.filterwarnings("ignore")
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    if args.wandb:
        wandb.init(
            project="VQA new",
            group=f"{args.model_name}-{args.dataset}",
            name= f"rank-{rank}",
            config=vars(args))
    
    directory_path = "./"
    extensions = ['.py', '.yaml', ".ipynb"]
    files_list = list_files(directory_path, extensions)
    directory = os.getcwd()
    if args.wandb:
        for filename in files_list:
            file_path = os.path.join(directory, filename)
            wandb.save(file_path, directory)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    if rank == 0:
        print(f"Dataset: {args.dataset}")
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
    
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    
    
    # if args.debug:
    #     val_loader = train_loader
    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    torch.cuda.set_device(rank)
    VQA_model  = call_model(args.model_name)
    
    if "hie" in args.model_name.lower():
        model = VQA_model(args = args,
                        question_vocab_size = train_dataset.token_size,
                        ans_vocab_size = train_dataset.vqa_output_dim,
                        question_type_map = train_dataset.question_type_map,
                        question_type_output_dim = train_dataset.question_type_output_dim
                        ).to(rank)
    else:
        model = VQA_model(args = args,
                        question_vocab_size = train_dataset.token_size,
                        ans_vocab_size = train_dataset.vqa_output_dim).to(rank)
    
    # optimizer = Adam(model.parameters(), lr=args.lr)
    # for param in model.parameters():
    #     param.requires_grad = True
    
    model = FSDP(model,
            auto_wrap_policy=my_auto_wrap_policy
            )
    for param in model.module.image_encoder.extractor.parameters():
        param.requires_grad = False
    
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    optimizer = Adam(model.parameters(), lr=args.lr)
    vqa_loss_fn = torch.nn.BCELoss(reduction='sum').to(rank)
    question_type_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum').to(rank)
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
        
     
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma, verbose = True)
    best_acc = 0
    test_best_result = None
    val_best_result = None
    stop_epoch = 0
    if rank == 0:
        print("Start training")
    trainer, validator, tester = call_engines(args)
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"------------------------- Epoch {str(epoch).zfill(3)} -------------------------")
        trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=train_sample)
        if epoch >= args.validation_epoch:
            if stop_epoch == args.early_stop:
                print("STOP TRAINING")
                break
            val_accuracy, val_result  = validator(model,loss_fn, rank, world_size, val_loader, epoch, args)
            
            if val_accuracy > best_acc:
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
            else:
                stop_epoch += 1
        scheduler.step()
    dist.barrier()
    if rank == 0 and args.wandb:
        val_predictions = pd.DataFrame(val_best_result)
        val_predictions.to_csv("val_predictions.csv", index=False)
        wandb.save("val_predictions.csv")
        
        test_predictions = pd.DataFrame(test_best_result)
        test_predictions.to_csv("test_predictions.csv", index=False)
    #     y_true = predictions['prediction']
    #     y_pred = predictions['target']
    #     # plot_confusion_matrix(y_true, y_pred)
        wandb.save("test_predictions.csv")
        # print('saving the model')
        # torch.save(model.state_dict(), "./checkpoints/bert-chatgptv1.pt")
        print('done!')
    if args.wandb:
        wandb.finish()

    cleanup()
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
    parser.add_argument('--epochs', type=int, default=100 , metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
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
    
    args = parser.parse_args()
    model_name = model_dict[args.model]
    
    args.model_name = model_name
    
    temp_result_path = f"./tem_results/{args.dataset}/{args.model_name}"
    result_path = f"./results/{args.dataset}/{args.model_name}"
    args.temp_result_path = temp_result_path
    args.result_path = result_path
    
    if os.path.exists(temp_result_path) and os.path.isdir(temp_result_path):
        shutil.rmtree(temp_result_path)
    os.makedirs(temp_result_path)
    
    if os.path.exists(result_path) and os.path.isdir(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)   
    if os.path.exists("./test_predictions.csv") and os.path.isdir("./test_predictions.csv"):
        shutil.rmtree("./test_predictions.csv")
    if os.path.exists("./val_predictions.csv") and os.path.isdir("./val_predictions.csv"):
        shutil.rmtree("./val_predictions.csv")
    
    config_model = yaml.safe_load(open(f'./config/models/{model_name}.yaml'))
    dataset_config = yaml.safe_load(open(f'./config/datasets/{args.dataset}.yaml'))
    vars(args).update(config_model)
    vars(args).update(dataset_config)
    

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)