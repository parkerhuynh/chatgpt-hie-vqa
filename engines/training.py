import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
import time

def trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    model.train()
    if rank == 0:
        print("    - Training")
    ddp_loss = torch.zeros(4).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        rnn_questions = batch['onehot_feature'].to(rank)
        images = batch['image'].to(rank)
        vqa_labels = batch['vqa_answer_label'].to(rank)
        bert_questions = batch['bert_input_ids'].to(rank)
        bert_attend_mask_questions = batch['bert_attention_mask'].to(rank)
        question_type_label = batch['question_type_label'].to(rank)
        
        
        if  "hie" in args.model_name:
            vqa_loss_fn, question_type_loss_fn = loss_fn
            
            vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
            vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
            question_type_loss = question_type_loss_fn(question_type_output,question_type_label)
            loss = vqa_loss + question_type_loss
            loss.backward()
            optimizer.step()
        else:
            vqa_loss_fn, _ = loss_fn
            vqa_output = model(images, rnn_questions)
            loss = vqa_loss_fn(vqa_output, vqa_labels)
            loss.backward()
            optimizer.step()
        
        vqa_pred_np = vqa_output.cpu().data.numpy()
        vqa_pred_argmax = np.argmax(vqa_pred_np, axis=1)
        vqa_indices = torch.tensor(vqa_pred_argmax)
        rows = torch.arange(vqa_labels.size(0))
        selected_values = vqa_labels[rows, vqa_indices]
        sum_selected_values = selected_values.sum()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += sum_selected_values.item()
        ddp_loss[2] += len(vqa_labels)
        
        if  "hie" in args.model_name:
            question_type_probabilities = F.softmax(question_type_output, dim=1)
            question_type_prediction = torch.argmax(question_type_probabilities, dim=1)
            question_type_prediction = question_type_prediction.cpu().numpy().tolist()
            question_type_label = question_type_label.cpu().numpy().tolist()
            ddp_loss[2] += sum(x == y for x, y in zip(question_type_prediction, question_type_label))
    
        if batch_idx % 50 == 0 and rank == 0:
            if args.wandb:
                wandb.log({"iter_loss": loss.item()/len(vqa_labels)})
            print(f'            - [{str(batch_idx).zfill(3)}/{str(len(train_loader)).zfill(3)}]:  loss: {(loss.item())/len(vqa_labels):.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[2]
        vqa_accuracy = ddp_loss[1] / ddp_loss[2]
        
        if  "hie" in args.model_name:
            question_type_accuracy = ddp_loss[2] / ddp_loss[2]
            print(f'        - Train Epoch {epoch}:  Average loss: {train_loss:.4f} | Question Type Accuracy: {ddp_loss[2]}/{int(ddp_loss[2])} ({100. * question_type_accuracy:.2f}%) | VQA Accuracy: {ddp_loss[1]}/{int(ddp_loss[2])} ({100. * vqa_accuracy:.2f}%) \n')
            if args.wandb:
                wandb.log({"epoch":epoch,
                    "train_loss": train_loss,
                    "train_vqa_accuracy": vqa_accuracy,
                    "train_question_type_accuracy": question_type_accuracy
                    })
            
        else:
            print(f'        - Train Epoch {epoch}:  Average loss: {train_loss:.4f} | VQA Accuracy: {ddp_loss[1]}/{int(ddp_loss[2])} ({100. * vqa_accuracy:.2f}%) \n')
            if args.wandb:
                wandb.log({"epoch":epoch,
                    "train_loss": train_loss,
                    "train_vqa_accuracy": vqa_accuracy
                    })



def normal_trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    model.train()
    if rank == 0:
        print("    - Training")
    ddp_loss = torch.zeros(3).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        rnn_questions = batch['onehot_feature'].to(rank)
        images = batch['image'].to(rank)
        vqa_labels = batch['vqa_answer_label'].to(rank)
        
        vqa_loss_fn, _ = loss_fn
        vqa_output = model(images, rnn_questions)
        loss = vqa_loss_fn(vqa_output, vqa_labels)
        loss.backward()
        optimizer.step()
        
        vqa_pred_np = vqa_output.cpu().data.numpy()
        vqa_pred_argmax = np.argmax(vqa_pred_np, axis=1)
        vqa_indices = torch.tensor(vqa_pred_argmax)
        rows = torch.arange(vqa_labels.size(0))
        selected_values = vqa_labels[rows, vqa_indices]
        sum_selected_values = selected_values.sum()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += sum_selected_values.item()
        ddp_loss[2] += len(vqa_labels)
        
    
        if batch_idx % 50 == 0 and rank == 0:
            if args.wandb:
                wandb.log({"iter_vqa_loss": loss.item()/len(vqa_labels)})
            print(f'            - [{str(batch_idx).zfill(3)}/{str(len(train_loader)).zfill(3)}]:  VQA loss: {(loss.item())/len(vqa_labels):.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[2]
        vqa_accuracy = ddp_loss[1] / ddp_loss[2]
        print(f'        - Train Epoch {epoch}:  Average loss: {train_loss:.4f} | VQA Accuracy: {ddp_loss[1]}/{int(ddp_loss[2])} ({100. * vqa_accuracy:.2f}%) \n')
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_vqa_loss": train_loss,
                "train_vqa_accuracy": vqa_accuracy
                })


def hie_trainer(args, model, rank, world_size, train_loader, optimizer, loss_fn, epoch, sampler=None):
    vqa_loss_fn, question_type_loss_fn = loss_fn
    model.train()
    if rank == 0:
        print("    - Training")
    ddp_loss = torch.zeros(6).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    
    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        rnn_questions = batch['onehot_feature'].to(rank)
        images = batch['image'].to(rank)
        vqa_labels = batch['vqa_answer_label'].to(rank)
        bert_questions = batch['bert_input_ids'].to(rank)
        bert_attend_mask_questions = batch['bert_attention_mask'].to(rank)
        question_type_label = batch['question_type_label'].to(rank)
        
        vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
        vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
        question_type_loss = question_type_loss_fn(question_type_output,question_type_label)
        
        loss = vqa_loss + question_type_loss
        loss.backward()
        optimizer.step()
        
        vqa_pred_np = vqa_output.cpu().data.numpy()
        vqa_pred_argmax = np.argmax(vqa_pred_np, axis=1)
        vqa_indices = torch.tensor(vqa_pred_argmax)
        rows = torch.arange(vqa_labels.size(0))
        selected_values = vqa_labels[rows, vqa_indices]
        sum_selected_values = selected_values.sum()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += vqa_loss.item()
        ddp_loss[2] += question_type_loss.item()
        ddp_loss[3] += sum_selected_values.item()
        question_type_probabilities = F.softmax(question_type_output, dim=1)
        question_type_prediction = torch.argmax(question_type_probabilities, dim=1)
        question_type_prediction = question_type_prediction.cpu().numpy().tolist()
        question_type_label = question_type_label.cpu().numpy().tolist()
        
        
        ddp_loss[4] += sum(x == y for x, y in zip(question_type_prediction, question_type_label))
        ddp_loss[5] += len(vqa_labels)
        
        if batch_idx % 50 == 0:
            if args.wandb:
                wandb.log({"iter_loss": loss.item()/len(vqa_labels), "vqa_loss": vqa_loss.item()/len(vqa_labels), "question_type_loss": question_type_loss.item()/len(vqa_labels)})
            if rank == 0:
                print(f'            - [{str(batch_idx).zfill(3)}/{str(len(train_loader)).zfill(3)}]:  loss: {(loss.item())/len(vqa_labels):.4f}|  vqa loss: {(vqa_loss.item())/len(vqa_labels):.4f}|  questionq type loss: {(question_type_loss.item())/len(vqa_labels):.4f}')
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[5]
        train_vqa_loss = ddp_loss[1] / ddp_loss[5]
        train_question_type_loss = ddp_loss[2] / ddp_loss[5]
        vqa_accuracy = ddp_loss[3] / ddp_loss[5]
        question_type_accuracy = ddp_loss[4] / ddp_loss[5]
        print(f'            - Train Epoch {epoch}:  Average loss: {train_loss:.4f}| VQA loss: {train_vqa_loss:.4f} | Question Type Loss: {train_question_type_loss:.4f} | Question Type Accuracy: {ddp_loss[4]}/{int(ddp_loss[5])} ({100. * question_type_accuracy:.2f}%) | VQA Accuracy: {ddp_loss[3]}/{int(ddp_loss[5])} ({100. * vqa_accuracy:.2f}%) \n')
        if args.wandb:
            wandb.log({"epoch":epoch,
                "train_loss": train_loss,
                "train_vqa_loss": train_vqa_loss,
                "train_question_type_loss": train_question_type_loss,
                "train_vqa_accuracy": vqa_accuracy,
                "train_question_type_accuracy": question_type_accuracy
                })