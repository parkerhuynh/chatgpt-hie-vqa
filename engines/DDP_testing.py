import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import numpy as np
import torch.nn.functional as F
import time
import json
import os
@torch.no_grad()
def normal_tester(model, rank, test_loader, args, epoch,  idx_to_vqa_ans, idx_to_question_type=None):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].cuda()
            
            vqa_output = model(images, rnn_questions)
            vqa_indices = torch.max(vqa_output, 1)[1].data # argmax
            
            if batch_idx % 50 ==0 and rank == 0:
                print(f'     - Testing | [{str(batch_idx).zfill(3)}/{str(len(test_loader)).zfill(3)}]')
            for ques_id, pres in zip(question_id, vqa_indices):
                item = {
                    "question_id": ques_id.item(),
                    "prediction": idx_to_vqa_ans[str(pres.item())],
                    }
                results.append(item)
        with open(os.path.join(args.temp_result_path, f"temp_result_epoch_{epoch}_rank_{rank}_test.json"), "w") as f:
            json.dump(results, f)
        del(results)
    

@torch.no_grad()
def hie_tester(model, rank, test_loader, args, epoch, idx_to_vqa_ans, idx_to_question_type):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].cuda()
            rnn_questions = batch['onehot_feature'].cuda()
            images = batch['image'].cuda()
            if "bert" in args.model_name.lower():
                bert_questions = batch['bert_input_ids'].cuda()
                bert_attend_mask_questions = batch['bert_attention_mask'].cuda()
            else:
                bert_questions = None
                bert_attend_mask_questions = None
            
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
        with open(os.path.join(args.temp_result_path, f"temp_result_epoch_{epoch}_rank_{rank}_test.json"), "w") as f:
            json.dump(results, f)
        del(results)