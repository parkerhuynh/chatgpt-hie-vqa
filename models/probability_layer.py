import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilityLayer(nn.Module):
    def __init__(self, question_type_map_dict, ans_vocab_size, question_type_output_di):
        super(ProbabilityLayer, self).__init__()
        print(question_type_map_dict)
        unique_question_types = set()
        for ans_id, q_types in question_type_map_dict.items():
            unique_question_types.update(q_types.keys())

        # Convert to list if necessary and sort (optional but helps in consistent indexing)
        question_type_ids = sorted(list(unique_question_types))
        answer_ids = sorted(list(question_type_map_dict.keys()))
        
        m = len(question_type_ids)  # Number of question types
        n = len(answer_ids)
        QuestionTypeMatrix = torch.zeros((m, n), dtype=torch.float32)
        
        question_type_to_index = {q_id: idx for idx, q_id in enumerate(question_type_ids)}
        answer_to_index = {a_id: idx for idx, a_id in enumerate(answer_ids)}

        # Populate the matrix
        for ans_id, q_type_percents in question_type_map_dict.items():
            ans_index = answer_to_index[ans_id]
            for q_type_id, percent in q_type_percents.items():
                if q_type_id in question_type_to_index:  # Ensure the question type is in our matrix
                    q_type_index = question_type_to_index[q_type_id]
                    QuestionTypeMatrix[q_type_index, ans_index] = percent
        
        self.register_buffer('QuestionTypeMatrix', QuestionTypeMatrix)
        
    def forward(self, question_type_output, vqa_output):
        question_type_output = F.softmax(question_type_output, dim=-1)
        current_device = question_type_output.device
        bz, _ = question_type_output.size()
        question_type_matrix = self.QuestionTypeMatrix.to(current_device).unsqueeze(0).expand(bz, -1, -1)
        
        question_type_output = question_type_output.unsqueeze(1)
        question_type_output = torch.bmm(question_type_output, question_type_matrix).squeeze(1)
        output = question_type_output*vqa_output
        return output