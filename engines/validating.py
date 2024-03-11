import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def normal_validator(model, loss_fn, rank, world_size, val_loader, epoch, args):
    vqa_loss_fn, _ = loss_fn
    if rank == 0:
        print("    - Validating")
    idx_to_vqa_ans = val_loader.dataset.idx_to_vqa_ans
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    accuracy = 0
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):  # Assuming question_id is part of your dataloader
            
            question_id = batch['question_id'].to(rank)
            rnn_questions = batch['onehot_feature'].to(rank)
            images = batch['image'].to(rank)
            vqa_labels = batch['vqa_answer_label'].to(rank)
            local_question_ids = question_id.cpu().numpy().tolist()
            
            vqa_output = model(images, rnn_questions)
            
            loss = vqa_loss_fn(vqa_output, vqa_labels)
            pred_np = vqa_output.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            indices = torch.tensor(pred_argmax)
            rows = torch.arange(vqa_labels.size(0))
            selected_values = vqa_labels[rows, indices]
            sum_selected_values = selected_values.sum()
            
            local_preds = pred_argmax
            for ques_id, pres in zip(local_question_ids, local_preds):
                item = {
                    "question_id": ques_id,
                    "prediction": idx_to_vqa_ans[str(pres)],
                    }
                results.append(item)
            
            # Loss calculation
            ddp_loss[0] += loss.item()
            ddp_loss[1] += sum_selected_values.item()
            ddp_loss[2] += len(vqa_labels)
            
            if batch_idx % 50 == 0 and rank == 0:
                print(f'            - [{str(batch_idx).zfill(3)}/{str(len(val_loader)).zfill(3)}]')
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[2]
        accuracy = ddp_loss[1] / ddp_loss[2]
        
        if rank == 0:
            print('        - Val Epoch  {}: Average VQA loss: {:.4f}, VQA Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                val_loss, ddp_loss[1], int(ddp_loss[2]),
                100. * accuracy))
            if args.wandb:
                wandb.log({"val_vqa_accuracy": accuracy,
                       "val_vqa_loss": val_loss,
                       "epoch":epoch})
        return accuracy, results


@torch.no_grad()
def hie_validator(model, loss_fn, rank, world_size, val_loader, epoch, args):
    vqa_loss_fn, question_type_loss_fn = loss_fn
    if rank == 0:
        print("    - Validating")
    idx_to_vqa_ans = val_loader.dataset.idx_to_vqa_ans
    idx_to_question_type = val_loader.dataset.idx_to_question_type
    model.eval()
    ddp_loss = torch.zeros(6).to(rank)
    accuracy = 0
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):  # Assuming question_id is part of your dataloader
            
            question_id = batch['question_id'].to(rank)
            rnn_questions = batch['onehot_feature'].to(rank)
            images = batch['image'].to(rank)
            vqa_labels = batch['vqa_answer_label'].to(rank)
            bert_questions = batch['bert_input_ids'].to(rank)
            bert_attend_mask_questions = batch['bert_attention_mask'].to(rank)
            local_question_ids = question_id.cpu().numpy().tolist()
            question_type_label = batch['question_type_label'].to(rank)
            vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
            
            vqa_loss = vqa_loss_fn(vqa_output, vqa_labels)
            question_type_loss = question_type_loss_fn(question_type_output,question_type_label)
            loss = vqa_loss + question_type_loss
            
            
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
            
            for ques_id, vqa_pres, question_type_pres in zip(local_question_ids, vqa_pred_argmax, question_type_prediction):
                item = {
                    "question_id": ques_id,
                    "vqa_prediction": idx_to_vqa_ans[str(vqa_pres)],
                    "question_prediction": idx_to_question_type[question_type_pres]
                    }
                results.append(item)
            
            if batch_idx % 50 == 0 and rank == 0:
                print(f'            - [{str(batch_idx).zfill(3)}/{str(len(val_loader)).zfill(3)}]')
                
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        val_loss = ddp_loss[0] / ddp_loss[5]
        val_vqa_loss = ddp_loss[1] / ddp_loss[5]
        val_question_type_loss = ddp_loss[2] / ddp_loss[5]
        vqa_accuracy = ddp_loss[3] / ddp_loss[5]
        question_type_accuracy = ddp_loss[4] / ddp_loss[5]
        
        if rank == 0:
            print(f'        - Val Epoch {epoch}:  Average loss: {val_loss:.4f}| VQA loss: {val_vqa_loss:.4f} | Question Type Loss: {val_question_type_loss:.4f} | Question Type Accuracy: {ddp_loss[4]}/{int(ddp_loss[5])} ({100. * question_type_accuracy:.2f}%) | VQA Accuracy: {ddp_loss[3]}/{int(ddp_loss[5])} ({100. * vqa_accuracy:.2f}%) \n')
            if args.wandb:
                wandb.log({"epoch":epoch,
                "val_loss": val_loss,
                "val_vqa_loss": val_vqa_loss,
                "val_question_type_loss": val_question_type_loss,
                "val_vqa_accuracy": vqa_accuracy,
                "val_question_type_accuracy": question_type_accuracy
                })
        return vqa_accuracy, results