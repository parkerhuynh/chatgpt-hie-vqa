import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
import time



def normal_trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    vqa_loss_fn, _ = loss_fn
    model.train()
    if rank == 0:
        print("    - Training")
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
        optimizer.step()
        
        _, vqa_indices = torch.max(vqa_output, dim=1)
        rows = torch.arange(vqa_labels.size(0))
        selected_values = vqa_labels[rows, vqa_indices]
        sum_selected_values = selected_values.sum()
        # if rank == 0:
        #     pred_np = vqa_output.cpu().data.numpy()
            
        #     print(batch['vqa_answer_str'])
            
        #     print("vqa_labels\n",vqa_labels)
        #     print("vqa_indices\n",vqa_indices)
        #     print("pred_argmax", np.argmax(pred_np, axis=1))
        #     print("selected_values\n",selected_values)
        #     print(sum_selected_values)
        #     print("-"*50)
            
        
        ddp_loss[0] += vqa_loss.item()
        ddp_loss[1] += sum_selected_values.item()
        ddp_loss[2] += len(vqa_labels)
        
        
        
        if args.wandb:
            wandb.log({
                "vqa_loss": vqa_loss.item(),
            })
            
        if rank == 0:
            print(f'   -[{batch_idx+1}/{len(train_loader)}]: VQA Loss: {vqa_loss.item()/len(vqa_labels):.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[2]
        vqa_accuracy = ddp_loss[1] / ddp_loss[2]
        print(f'- Train Epoch {epoch}:  Average loss: {train_loss:.4f} | VQA Accuracy: {ddp_loss[1]}/{int(ddp_loss[2])} ({100. * vqa_accuracy:.2f}%) \n')
        
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_vqa_loss": train_loss,
                "train_vqa_accuracy": vqa_accuracy,
                })

def hie_trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    if sampler:
        train_loader.sampler.set_epoch(epoch)
    vqa_loss_fn, question_type_loss_fn = loss_fn
    model.train()
    if rank == 0:
        print("    - Training")
    ddp_loss = torch.zeros(6).to(rank)
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        images = batch['image'].cuda()
        rnn_questions = batch['onehot_feature'].cuda()
        vqa_labels = batch['vqa_answer_label'].cuda()
        bert_questions = batch['bert_input_ids'].cuda()
        bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
        question_type_label = batch['question_type_label'].cuda()
        
        vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
        vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
        question_type_loss = question_type_loss_fn(question_type_output, question_type_label)
        
        loss = 0.8*vqa_loss + 0.2*question_type_loss
        loss.backward()
        optimizer.step()
        
        vqa_pred_argmax_prob, vqa_indices = torch.max(vqa_output, dim=1)
        rows = torch.arange(vqa_labels.size(0))
        selected_values = vqa_labels[rows, vqa_indices]
        sum_selected_values = selected_values.sum()
        
        _, question_type_predictions = torch.max(question_type_output.data, 1)
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += vqa_loss.item()
        ddp_loss[2] += question_type_loss.item()
        ddp_loss[3] += sum_selected_values.item()
        ddp_loss[4] += sum(x == y for x, y in zip(question_type_predictions, question_type_label))
        ddp_loss[5] += len(vqa_labels)
        
        
        
        if args.wandb:
            wandb.log({
                "total_loss": loss.item(),
                "vqa_loss": vqa_loss.item(),
                "question_type_loss": question_type_loss.item(),
            })
            
        if rank == 0:
            print(f'   -[{batch_idx+1}/{len(train_loader)}]: Loss: {loss.item()/len(vqa_labels):.4f}, VQA Loss: {vqa_loss.item()/len(vqa_labels):.4f}, Question Type Loss: {question_type_loss.item()/len(vqa_labels):.4f}')
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[5]
        train_vqa_loss = ddp_loss[1] / ddp_loss[5]
        train_question_type_loss = ddp_loss[2] / ddp_loss[5]
        vqa_accuracy = ddp_loss[3] / ddp_loss[5]
        question_type_accuracy = ddp_loss[4] / ddp_loss[5]
        print(f'- Train Epoch {epoch}:  Average loss: {train_loss:.4f}| VQA loss: {train_vqa_loss:.4f} | Question Type Loss: {train_question_type_loss:.4f} | Question Type Accuracy: {ddp_loss[4]}/{int(ddp_loss[5])} ({100. * question_type_accuracy:.2f}%) | VQA Accuracy: {round(ddp_loss[3], 2)}/{int(ddp_loss[5])} ({100. * vqa_accuracy:.2f}%) \n')
        
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_loss": train_loss,
                "train_vqa_loss": train_vqa_loss,
                "train_question_type_loss": train_question_type_loss,
                "train_vqa_accuracy": vqa_accuracy,
                "train_question_type_accuracy": question_type_accuracy
                })