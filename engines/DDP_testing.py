import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def normal_tester(model, rank, world_size, test_loader):
    if rank == 0:
        print("    - Testing")
    idx_to_vqa_ans = test_loader.dataset.idx_to_vqa_ans
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].cuda()
            
            vqa_output = model(images, rnn_questions)
            _, pred = torch.max(vqa_output.data, 1)
            pred_np = vqa_output.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            local_preds = pred_argmax
            
            local_question_ids = question_id.cpu().numpy().tolist()
            
            if  rank == 0:
                print(f'   - [{str(batch_idx).zfill(3)}/{str(len(test_loader)).zfill(3)}]')
            for ques_id, pres in zip(local_question_ids, local_preds):
                item = {
                    "question_id": ques_id,
                    "prediction": idx_to_vqa_ans[str(pres)],
                    }
                results.append(item)
        return results
    

@torch.no_grad()
def hie_tester(model, rank, world_size, test_loader):
    if rank == 0:
        print("    - Testing")
    idx_to_vqa_ans = test_loader.dataset.idx_to_vqa_ans
    idx_to_question_type = test_loader.dataset.idx_to_question_type
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].to(rank)
            bert_questions = batch['bert_input_ids'].cuda()
            bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
            
            vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
            vqa_pred_np = vqa_output.cpu().data.numpy()
            vqa_pred_argmax = np.argmax(vqa_pred_np, axis=1)
            
            question_type_probabilities = F.softmax(question_type_output, dim=1)
            question_type_prediction = torch.argmax(question_type_probabilities, dim=1)
            question_type_prediction = question_type_prediction.cpu().numpy().tolist()
            
            local_question_ids = question_id.cpu().numpy().tolist()
            
            if batch_idx % 50 == 0 and rank == 0:
                print(f'            - [{str(batch_idx).zfill(3)}/{str(len(test_loader)).zfill(3)}]')
            for ques_id, vqa_pres, question_type_pres in zip(local_question_ids, vqa_pred_argmax, question_type_prediction):
                item = {
                    "question_id": ques_id,
                    "vqa_prediction": idx_to_vqa_ans[str(vqa_pres)],
                    "question_prediction": idx_to_question_type[question_type_pres]
                }
                results.append(item)
        return results