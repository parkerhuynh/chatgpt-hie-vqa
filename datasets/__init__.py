from datasets.vqa_dataset_simpsons import VQADataset as vqa_dataset_simpsons
from datasets.vqa_dataset_vqav2 import VQADataset as vqa_dataset_vqav2


def create_vqa_datasets(args, rank):
    dataset_fn = globals()[f"vqa_dataset_{args.dataset}"]
    
    train_dataset = dataset_fn(args, "train", rank)
    val_dataset = dataset_fn(args, "val", rank)
    test_dataset = dataset_fn(args, "test", rank)
    
    return [train_dataset, val_dataset, test_dataset]