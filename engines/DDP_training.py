import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
import time
from loss_fn import compute_score_with_logits_paddingremoved


def normal_trainer(args, model, rank, world_size, train_loader, optimizers, loss_fn, epoch, sampler=None):
    optimizer, _ = optimizers
    vqa_loss_fn, _ = loss_fn
    model.train()
    if sampler:
        train_loader.sampler.set_epoch(epoch)
    ddp_loss = torch.zeros(3).to(rank)
    
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        images = batch['image'].cuda()
        rnn_questions = batch['onehot_feature'].cuda()
        vqa_labels = batch['vqa_answer_label'].cuda()
        
        
        vqa_output = model(images, rnn_questions)
        vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
        vqa_loss.backward()
        print_vqa_loss = vqa_loss.item()* images.size(0)
        optimizer.step()
        
        batch_score , batch_count= compute_score_with_logits_paddingremoved(vqa_output, vqa_labels)
    
        ddp_loss[0] += print_vqa_loss
        ddp_loss[1] += batch_score.item() 
        ddp_loss[2] += batch_count
        
        if args.wandb:
            wandb.log({
                "vqa_loss": print_vqa_loss,
            })
            
        if batch_idx % 50 == 0 and rank == 0:
            print(f'     - Training | [{batch_idx+1}/{len(train_loader)}] | VQA Loss: {print_vqa_loss:.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / (batch_idx+1)
        vqa_accuracy = ddp_loss[1] / ddp_loss[2]
        print(
            f'==> Train Epoch {epoch} | '
            f'Average loss: {train_loss:.4f} | '
            f'VQA Accuracy: {round(ddp_loss[1].item(), 2)}/{int(ddp_loss[2])} '
            f'({100. * vqa_accuracy:.2f}%) \n'
        )
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_vqa_loss": train_loss,
                "train_vqa_accuracy": vqa_accuracy,
                })

def hie_trainer(args, model, rank, world_size, train_loader, optimizers, loss_fn, epoch, sampler=None):
    if sampler:
        train_loader.sampler.set_epoch(epoch)
    optimizer_for_question_type, optimizer_for_rest = optimizers
    vqa_loss_fn, question_type_loss_fn = loss_fn
    model.train()

    ddp_loss = torch.zeros(6).to(rank)
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer_for_question_type.zero_grad()
        optimizer_for_rest.zero_grad()
        
        images = batch['image'].cuda()
        rnn_questions = batch['onehot_feature'].cuda()
        vqa_labels = batch['vqa_answer_label'].cuda()
        bert_questions = batch['bert_input_ids'].cuda()
        bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
        question_type_label = batch['question_type_label'].cuda()
        
        vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
        vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
        question_type_loss = question_type_loss_fn(question_type_output, question_type_label)
        
        loss = args.loss_weight*vqa_loss  + (1-args.loss_weight)*question_type_loss
        
        loss.backward()
        optimizer_for_question_type.step()
        optimizer_for_rest.step()
        
        batch_score , batch_count= compute_score_with_logits_paddingremoved(vqa_output, vqa_labels)
             
        _, question_type_predictions = torch.max(question_type_output.data, 1)
    
        ddp_loss[0] += loss.item()* images.size(0)
        ddp_loss[1] += args.loss_weight*vqa_loss.item()* images.size(0)
        ddp_loss[2] += (1-args.loss_weight)*question_type_loss.item()* images.size(0)
        ddp_loss[3] += batch_score.item()
        ddp_loss[4] += sum(x == y for x, y in zip(question_type_predictions, question_type_label))
        ddp_loss[5] += batch_count
        
        
        
        if args.wandb:
            wandb.log({
                "total_loss": loss.item()* images.size(0),
                "vqa_loss": args.loss_weight*vqa_loss.item()* images.size(0),
                "question_type_loss": (1 - args.loss_weight)*question_type_loss.item()* images.size(0),
            })

        if batch_idx % 50 == 0 and rank == 0:
            print(
                f'     - Training | [{batch_idx+1}/{len(train_loader)}] | '
                f'Loss: {loss.item() * images.size(0):.4f} | '
                f'VQA Loss: {args.loss_weight * vqa_loss.item() * images.size(0):.4f} | '
                f'Question Type Loss: {(1-args.loss_weight) * question_type_loss.item() * images.size(0):.4f}'
            )
            
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    total_batch = batch_idx+1
    if rank == 0:
        train_loss = ddp_loss[0] / total_batch
        train_vqa_loss = ddp_loss[1] / total_batch
        train_question_type_loss = ddp_loss[2] / total_batch
        vqa_accuracy = ddp_loss[3] / ddp_loss[5]
        question_type_accuracy = ddp_loss[4] / ddp_loss[5]
        print(
            f'==> Train | Epoch {epoch}: | '
            f'Average loss: {train_loss:.4f} | '
            f'VQA loss: {train_vqa_loss:.4f} | '
            f'Question Type Loss: {train_question_type_loss:.4f} | '
            f'Question Type Accuracy: {ddp_loss[4]}/{int(ddp_loss[5])} '
            f'({100. * question_type_accuracy:.2f}%) | '
            f'VQA Accuracy: {round(ddp_loss[3].item(), 2)}/{int(ddp_loss[5])} '
            f'({100. * vqa_accuracy:.2f}%) \n'
        )
        
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_loss": train_loss,
                "train_vqa_loss": train_vqa_loss,
                "train_question_type_loss": train_question_type_loss,
                "train_vqa_accuracy": vqa_accuracy,
                "train_question_type_accuracy": question_type_accuracy
                })