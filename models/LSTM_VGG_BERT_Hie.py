import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn.functional as F
    
class ImageEncoder(nn.Module):

    def __init__(self, output_size=1024):
        super(ImageEncoder, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

    def forward(self, image):
        image = self.extractor(image)
        image_embedding = self.fflayer(image)
        norm = image_embedding.norm(p=2, dim=1, keepdim=True)
        image_embedding = image_embedding.div(norm)
        return image_embedding

class QuestionEmbedding(nn.Module):
    def __init__(self, word_embedding_size, hidden_size):
        super(QuestionEmbedding, self).__init__()
        self.lstm = nn.LSTM(word_embedding_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1024)

    def forward(self, input_data):
        _, (hidden, _) = self.lstm(input_data)
        last_hidden = hidden[-1]
        embedding = self.fc(last_hidden)
        return embedding
    

class QuestionType(nn.Module):
    def __init__(self, question_type_output_dim):
        super(QuestionType, self).__init__()
        self.BERT_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=question_type_output_dim)

    def forward(self, question_bert, bert_attend_mask_questions):
        output = self.BERT_classification(question_bert, attention_mask=bert_attend_mask_questions)
        logits = output.logits
        return logits
    

class CustomLayer(nn.Module):
    def __init__(self, question_type_map_dict, ans_vocab_size, question_type_output_di):
        super(CustomLayer, self).__init__()
        
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
    
    
    
class LSTM_VGG_BERT_Hie(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size, question_type_map, question_type_output_dim):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(LSTM_VGG_BERT_Hie, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(output_size = args.image_feature_output)

        self.WordEmbedding = nn.Embedding(question_vocab_size, args.word_embedding)
        self.QuestionEncoder = QuestionEmbedding(
            word_embedding_size = args.word_embedding,
            hidden_size = args.rnn_hidden_size
            )
        
        
        self.QuestionType = QuestionType(question_type_output_dim)
        
        self.mlp_1 = nn.Sequential(
                nn.Linear(args.image_feature_output, 1000),
                nn.Dropout(p=0.5, inplace=True),
                nn.ReLU(),
                nn.Linear(1000, ans_vocab_size))
        
        self.CustomLayer = CustomLayer(question_type_map, ans_vocab_size, question_type_output_dim)
        
        self.mlp_2 = nn.Sequential(
                nn.Linear(ans_vocab_size, 1000),
                nn.Dropout(p=0.5, inplace=True),
                nn.ReLU(),
                nn.Linear(1000, ans_vocab_size))
        
    def forward(self, image, question_rnn, question_bert, bert_attend_mask_questions):
        image = self.image_encoder(image)
        question_rnn = self.WordEmbedding(question_rnn)
        question_rnn = self.QuestionEncoder(question_rnn)
        combine  = torch.mul(image,question_rnn)        
        vqa_output = self.mlp_1(combine)
        
        question_type_output = self.QuestionType(question_bert, bert_attend_mask_questions)
        vqa_output = self.CustomLayer(question_type_output, vqa_output)
        vqa_output = self.mlp_2(vqa_output)
        # output = torch.sigmoid(output)
        return vqa_output, question_type_output
        # return question_type_output
