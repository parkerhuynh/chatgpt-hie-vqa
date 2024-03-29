import json
import os
import torch.distributed as dist
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import numpy as np
import wandb
import torch

def read_json(rpath: str):
    with open(rpath, "r") as f:
        result = json.load(f)
    return result
def collect_result(rank, epoch, split, args):
    dist.barrier()
    result = []
    if rank == 0:
        for rank_id in range(dist.get_world_size()):
            temp_result_path_i = os.path.join(args.temp_result_path, f"temp_result_epoch_{epoch}_rank_{rank_id}_{split}.json")
            result += read_json(temp_result_path_i)

        result_new = []
        id_list = set()
        for res in result:
            if res["question_id"] not in id_list:
                id_list.add(res["question_id"])
                result_new.append(res)
        result = result_new
        result_path = os.path.join(args.result_path, f"epoch_{epoch}_{split}.json")
        json.dump(result, open(result_path, 'w'), indent=4)
        print(f'==> {split} | total number of {split} set: {len(result)} | saving {result_path} |')
    dist.barrier()
    return result

def plot_confusion_matrix(y_true, y_pred):
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=y_true.unique())
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(30, 12))
    
    # Regular confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', 
                xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    
    # Normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', 
                xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image = Image.open(buffer)
    image_array = np.array(image)
    wandb.log({"Confusion Matrix": wandb.Image(image_array)})
    
    # Close the plot
    plt.close()
    
def list_files(directory, extensions):
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, filename)
                if "wandb" not in full_path:
                    matches.append(os.path.join(root, filename))
    return matches



#############################################################################################

def setup_for_printing_everywhere():
    """
    This function ensures printing is enabled on all devices in a distributed setting.
    """
    import builtins as __builtin__
    # Save the original built-in print function
    builtin_print = __builtin__.print

    # Override the built-in print with a new version that always prints
    def print(*args, **kwargs):
        # Call the original saved print function
        builtin_print(*args, **kwargs)

    # Replace the built-in print function with the overridden version
    __builtin__.print = print
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_printing_everywhere()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


  
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()