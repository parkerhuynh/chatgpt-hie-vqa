import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataset.dataset import QuestionDataset
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
from model import VQA as VQA_model
import random
from torch.optim import Adam
os.environ["#wandb_START_METHOD"] = "thread"
import shutil
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(3).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        rnn_questions = batch['onehot_feature'].to(rank)
        images = batch['image'].to(rank)
        vqa_labels = batch['vqa_answer_label'].to(rank)
        
        # print(images.size())
        # print(rnn_questions.size())
        # print(vqa_labels.size())
        # print(rnn_questions.size())
        # print("-"*100)
        
        output = model(images, rnn_questions)
        
        _, pred = torch.max(output.data, 1)
        
        loss = loss_fn(output, vqa_labels)
        
        loss.backward()
        optimizer.step()
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (pred == vqa_labels).sum().item()
        ddp_loss[2] += len(vqa_labels)
        
        if batch_idx % 50 == 0 and rank == 1:
            wandb.log({"iter_loss": loss.item()/len(vqa_labels)})
            print(f'Train Epoch {epoch} [{batch_idx}/{len(train_loader)}]:  loss: {(loss.item())/len(vqa_labels):.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[2]
        accuracy = ddp_loss[1] / ddp_loss[2]
        print('Train Epoch {}:  Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                train_loss, int(ddp_loss[1]), int(ddp_loss[2]),
                100. * accuracy))
        wandb.log({"epoch":epoch,
                   "train_loss": train_loss,
                   "train_accuracy": accuracy
                   })
        
    
def test(model, loss_fn, rank, world_size, test_loader, epoch):
    idx_to_vqa_ans = test_loader.dataset.idx_to_vqa_ans
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    results = []
    ids = set()
    ids_list = []
    accuracy = 0
    with torch.no_grad():
        for batch in test_loader:  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].to(rank)
            rnn_questions = batch['onehot_feature'].to(rank)
            images = batch['image'].to(rank)
            vqa_labels = batch['vqa_answer_label'].to(rank)
            
            output = model(images, rnn_questions)
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, vqa_labels)
            local_preds = pred.cpu().numpy().tolist()
            
            local_question_ids = question_id.cpu().numpy().tolist()
            ids.update(set(local_question_ids))
            ids_list += local_question_ids
            local_targets = vqa_labels.cpu().numpy().tolist()
            # Loss calculation
            ddp_loss[0] += loss.item()
            ddp_loss[1] += pred.eq(vqa_labels.view_as(pred)).sum().item()
            ddp_loss[2] += len(vqa_labels)
            
            for ques_id, pres, target in zip(local_question_ids, local_preds, local_targets):
                item = {
                    "question_id": ques_id,
                    "prediction": idx_to_vqa_ans[str(pres)],
                    "target": idx_to_vqa_ans[str(target)]
                    }
                results.append(item)
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            accuracy = ddp_loss[1] / ddp_loss[2]
            print('Test Epoch  {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
                100. * accuracy))
            wandb.log({"val_accuracy": accuracy,
                       "val_loss": test_loss,
                       "epoch":epoch})
        return accuracy, results

        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    
    wandb.init(
            project="VQA new",
            group="VQA",
            name= f"Normal VQA",
            config=vars(args))
    
    directory_path = "./"
    extensions = ['.py', '.yaml', ".ipynb"]
    files_list = list_files(directory_path, extensions)
    directory = os.getcwd()
    for filename in files_list:
        file_path = os.path.join(directory, filename)
        wandb.save(file_path, directory)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    dataset1 = QuestionDataset(args, "train")
    dataset2 = QuestionDataset(args, "val")
    if rank == 0:
        print(f"Number of Traning Sample: {len(dataset1)}")
        print(f"Number of Validation Sample: {len(dataset2)}")
        print(f"Question Vocabulary Size: {dataset1.token_size}")

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size,shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    
    
    # if args.debug:
    #     test_loader = train_loader
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    torch.cuda.set_device(rank)

    model = VQA_model(args = args,
                      question_vocab_size = dataset1.token_size,
                      ans_vocab_size = dataset1.vqa_output_dim).to(rank)
    model = FSDP(model,
            auto_wrap_policy=my_auto_wrap_policy)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss(reduction  = "sum")
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        untrainable_params = total_params - trainable_params
        
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Untrainable Parameters: {untrainable_params}")

    best_acc = 0
    best_result = None
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=sampler1)
        val_accuracy, results  = test(model,loss_fn, rank, world_size, test_loader, epoch)
        final_result = collect_result(results, rank, epoch)
        if val_accuracy > best_acc and rank == 0:
            best_acc = val_accuracy
            best_result = final_result
            wandb.log({"best_accuracy": best_acc})
        # scheduler.step()
    dist.barrier()
    if rank == 0:
        predictions = pd.DataFrame(best_result)
        predictions.to_csv("predictions.csv", index=False)
        y_true = predictions['prediction']
        y_pred = predictions['target']
        plot_confusion_matrix(y_true, y_pred)
        wandb.save("predictions.csv")
        # print('saving the model')
        # torch.save(model.state_dict(), "./checkpoints/bert-chatgptv1.pt")
        print('done!')
    dist.barrier()
    wandb.finish()

    cleanup()
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.01, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='For Debuging')

    if os.path.exists("./temp_result") and os.path.isdir("./temp_result"):
        shutil.rmtree("./temp_result")
    os.makedirs("./temp_result")
    args = parser.parse_args()
    config = yaml.safe_load(open(f'./config.yaml'))
    vars(args).update(config)

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)