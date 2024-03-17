import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
from loss_fn import compute_score_with_logits_paddingremoved
import time
@torch.no_grad()
def normal_validator(model, loss_fn, rank, world_size, val_loader, epoch, args):
    epoch_start_time = time.time()
    model.eval()
    vqa_loss_fn, _ = loss_fn
    idx_to_vqa_ans = val_loader.dataset.idx_to_vqa_ans
    
    ddp_loss = torch.zeros(3).to(rank)
    accuracy = 0
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):  # Assuming question_id is part of your dataloader
            
            question_id = batch['question_id'].cuda()
            images = batch['image'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            vqa_labels = batch['vqa_answer_label'].cuda()
            
            vqa_output = model(images, rnn_questions)
            vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
            
            print_vqa_loss = vqa_loss.item()* images.size(0)
            batch_score , batch_count= compute_score_with_logits_paddingremoved(vqa_output, vqa_labels)

            logits = torch.max(vqa_output, 1)[1].data # argmax
            ddp_loss[0] += print_vqa_loss
            ddp_loss[1] += batch_score.item() 
            ddp_loss[2] += batch_count
            
            
            for ques_id, pres in zip(question_id, logits):
                item = {
                    "question_id": ques_id.item(),
                    "prediction": idx_to_vqa_ans[str(pres.item())],
                    }
                results.append(item)
            
            if rank == 0:
                print(f'     - Validating [{str(batch_idx).zfill(4)}/{str(len(val_loader)).zfill(4)}]')

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / (batch_idx+1)
    accuracy = ddp_loss[1] / ddp_loss[2]
    
    if batch_idx % 50 == 0 and rank == 0:
        epoch_end_time = time.time()
        epoch_elapsed_time = round(epoch_end_time - epoch_start_time, 4)
        print(f'==> Validation | Epoch  {epoch} | Average VQA loss: {val_loss:.4f} | VQA Accuracy: {round(ddp_loss[1].item(),2)}/{int(ddp_loss[2])} ({(100. * accuracy):.2f}%) | Running time {epoch_elapsed_time}')
        if args.wandb:
            wandb.log({"val_vqa_accuracy": accuracy,
                    "val_vqa_loss": val_loss,
                    "epoch":epoch})
    return accuracy, results


@torch.no_grad()
def hie_validator(model, loss_fn, rank, world_size, val_loader, epoch, args):
    epoch_start_time = time.time()
    model.eval()
    vqa_loss_fn, question_type_loss_fn = loss_fn
    idx_to_vqa_ans = val_loader.dataset.idx_to_vqa_ans
    idx_to_question_type = val_loader.dataset.idx_to_question_type
    
    ddp_loss = torch.zeros(6).to(rank)
    accuracy = 0
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):  # Assuming question_id is part of your dataloader
            start_time = time.time()
            question_id = batch['question_id'].cuda()
            images = batch['image'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            vqa_labels = batch['vqa_answer_label'].cuda()
            bert_questions = batch['bert_input_ids'].cuda()
            bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
            question_type_label = batch['question_type_label'].cuda()
            
            
            vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
            vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
            question_type_loss = question_type_loss_fn(question_type_output, question_type_label)
            vqa_indices = torch.max(vqa_output, 1)[1].data # argmax
            
            loss = args.loss_weight*vqa_loss  + (1-args.loss_weight)*question_type_loss
            batch_score , batch_count= compute_score_with_logits_paddingremoved(vqa_output, vqa_labels)
                
            _, question_type_predictions = torch.max(question_type_output.data, 1)
            end_time = time.time()
            elapsed_time = round(end_time - start_time,4)
            
            
            ddp_loss[0] += loss.item()* images.size(0)
            ddp_loss[1] += args.loss_weight*vqa_loss.item()* images.size(0)
            ddp_loss[2] += (1-args.loss_weight)*question_type_loss.item()* images.size(0)
            ddp_loss[3] += batch_score.item()
            ddp_loss[4] += sum(x == y for x, y in zip(question_type_predictions, question_type_label))
            ddp_loss[5] += batch_count
            
            for ques_id, vqa_pres, question_type_pres in zip(question_id, vqa_indices, question_type_predictions):
                item = {
                    "question_id": ques_id.item(),
                    "vqa_prediction": idx_to_vqa_ans[str(vqa_pres.item())],
                    "question_prediction": idx_to_question_type[question_type_pres.item()]
                    }
                results.append(item)
            
            if batch_idx % 50 == 0 and rank == 0:
                print(f'     - Validating | [{str(batch_idx).zfill(3)}/{str(len(val_loader)).zfill(3)}]|  Running time: {elapsed_time} seconds')
        total_batch = batch_idx+1
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / total_batch
        val_vqa_loss = ddp_loss[1] / total_batch
        val_question_type_loss = ddp_loss[2] / total_batch
        vqa_accuracy = ddp_loss[3] / ddp_loss[5]
        question_type_accuracy = ddp_loss[4] / ddp_loss[5]
        
        epoch_end_time = time.time()
        epoch_elapsed_time = round(epoch_end_time - epoch_start_time, 4)
        if rank == 0:
            print(
                f'==> Validation | Epoch {epoch}:  Average loss: {val_loss:.4f} | '
                f'VQA loss: {val_vqa_loss:.4f} | Question Type Loss: {val_question_type_loss:.4f} | '
                f'Question Type Accuracy: {ddp_loss[4]}/{int(ddp_loss[5])} '
                f'({100. * question_type_accuracy:.2f}%) | '
                f'VQA Accuracy: {round(ddp_loss[3].item(),2)}/{int(ddp_loss[5])} '
                f'({100. * vqa_accuracy:.2f}%) | '
                f'Running Time : {epoch_elapsed_time} seconds\n'
                
            )
            if args.wandb:
                wandb.log({"epoch":epoch,
                "val_loss": val_loss,
                "val_vqa_loss": val_vqa_loss,
                "val_question_type_loss": val_question_type_loss,
                "val_vqa_accuracy": vqa_accuracy,
                "val_question_type_accuracy": question_type_accuracy
                })
        return vqa_accuracy, results