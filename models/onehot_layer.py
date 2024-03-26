import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotLayer(nn.Module):
    def __init__(self, question_type_map_dict, ans_vocab_size, question_type_output_dim):
        super(OneHotLayer, self).__init__()
        QuestionTypeMatrix = torch.zeros((question_type_output_dim, ans_vocab_size), dtype=torch.float32)
        for answer_index, question_types in question_type_map_dict.items():
            answer_index = int(answer_index)
            for question_type in question_types:
                QuestionTypeMatrix[question_type, answer_index] = 1
        self.register_buffer('QuestionTypeMatrix', QuestionTypeMatrix)
            

    def forward(self, question_type_output, vqa_output):
        question_type_output = F.softmax(question_type_output, dim=-1)
        current_device = question_type_output.device
        bz, _ = question_type_output.size()
        question_type_matrix = self.QuestionTypeMatrix.to(current_device).unsqueeze(0).expand(bz, -1, -1)
        question_type_output = question_type_output.unsqueeze(1)
        # print("question_type_output", question_type_output.size())
        # print("question_type_matrix", question_type_matrix.size())
        question_type_output = torch.bmm(question_type_output, question_type_matrix).squeeze(1)
        # question_type_output = question_type_output.max(dim=1)[0] 

        # print("output", question_type_output.size())
        output = question_type_output*vqa_output
        return output