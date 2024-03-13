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

def read_json(rpath: str):
    result = []
    with open(rpath, 'rt') as f:
        for line in f:
            result.append(json.loads(line.strip()))

    return result
def collect_result(result, rank, epoch, split, args):
    main_temp_result_path = os.path.join(args.temp_result_path, f"temp_result_epoch_{epoch}_rank_{rank}_{split}.json")
    with open(main_temp_result_path, 'wt') as f:
        for res in result:
            f.write(json.dumps(res) + '\n')
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
        print(f"saving {result_path}")

    dist.barrier()
    if rank == 0:
        print(f"total number of {split} set: {len(result)}")
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