import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
import time
@torch.no_grad()
def normal_tester(model, rank, world_size, test_loader):
    idx_to_vqa_ans = test_loader.dataset.idx_to_vqa_ans
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].cuda()
            
            vqa_output = model(images, rnn_questions)
            vqa_indices = torch.max(vqa_output, 1)[1].data # argmax
            
            if  rank == 0:
                print(f'     - Testing | [{str(batch_idx).zfill(3)}/{str(len(test_loader)).zfill(3)}]')
            for ques_id, pres in zip(question_id, vqa_indices):
                item = {
                    "question_id": ques_id.item(),
                    "prediction": idx_to_vqa_ans[str(pres.item())],
                    }
                results.append(item)
        return results
    

@torch.no_grad()
def hie_tester(model, rank, world_size, test_loader):
    idx_to_vqa_ans = test_loader.dataset.idx_to_vqa_ans
    idx_to_question_type = test_loader.dataset.idx_to_question_type
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].cuda()
            bert_questions = batch['bert_input_ids'].cuda()
            bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
            
            vqa_output, question_type_output = model(images, rnn_questions, bert_questions, bert_attend_mask_questions)
            vqa_indices = torch.max(vqa_output, 1)[1].data # argmax
                
            _, question_type_predictions = torch.max(question_type_output.data, 1)
            if batch_idx % 50 == 0 and rank == 0:
                print(f'     - Testing | [{str(batch_idx).zfill(3)}/{str(len(test_loader)).zfill(3)}]')
            for ques_id, vqa_pres, question_type_pres in zip(question_id, vqa_indices, question_type_predictions):
                item = {
                    "question_id": ques_id.item(),
                    "vqa_prediction": idx_to_vqa_ans[str(vqa_pres.item())],
                    "question_prediction": idx_to_question_type[question_type_pres.item()]
                }
                results.append(item)
        return results