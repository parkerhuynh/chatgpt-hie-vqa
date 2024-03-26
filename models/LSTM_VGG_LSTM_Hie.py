import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn.functional as F
from models.onehot_layer import OneHotLayer
from models.probability_layer import ProbabilityLayer


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
    def __init__(self, word_embedding_size, hidden_dim, question_type_output_dim):
        super(QuestionType, self).__init__()
        self.lstm = nn.LSTM(word_embedding_size, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, question_type_output_dim)

    def forward(self, question_embedded):
        _, (hidden, _) = self.lstm(question_embedded)
        # hidden is of shape [num_layers, batch_size, hidden_dim]
        # We take the last layer's hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        logits = self.fc(last_hidden)
        return logits
    
class LSTM_VGG_LSTM_Hie(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size, question_type_map, question_type_output_dim):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(LSTM_VGG_LSTM_Hie, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(output_size = args.image_feature_output)

        self.WordEmbedding = nn.Embedding(question_vocab_size, args.word_embedding)
        self.QuestionEncoder = QuestionEmbedding(
            word_embedding_size = args.word_embedding,
            hidden_size = args.rnn_hidden_size
            )
        
        
        self.QuestionType = QuestionType(args.word_embedding, args.rnn_hidden_size, question_type_output_dim)
        
        self.mlp_1 = nn.Sequential(
                nn.Linear(args.image_feature_output, 1000),
                nn.Dropout(p=0.5, inplace=True),
                nn.ReLU(),
                nn.Linear(1000, ans_vocab_size))
        
        if args.layer_name == "onehot":
            print("onehot")
            self.CustomLayer = OneHotLayer(question_type_map, ans_vocab_size, question_type_output_dim)
        else:
            print("probability")
            self.CustomLayer = ProbabilityLayer(question_type_map, ans_vocab_size, question_type_output_dim)
        if args.architecture == 1:
            self.mlp_2 = nn.Sequential(
                nn.Linear(ans_vocab_size, 1000),
                nn.Dropout(p=0.5, inplace=True),
                nn.ReLU(),
                nn.Linear(1000, ans_vocab_size)
            )
        
    def forward(self, image, question_rnn_input, question_bert=None, bert_attend_mask_questions=None):
        image = self.image_encoder(image)
        word_emb = self.WordEmbedding(question_rnn_input)
        question_rnn = self.QuestionEncoder(word_emb)
        combine  = torch.mul(image,question_rnn)        
        vqa_output = self.mlp_1(combine)
        
        question_type_output = self.QuestionType(word_emb)
        vqa_output = self.CustomLayer(question_type_output, vqa_output)
        if self.args.architecture == 1:
            vqa_output = self.mlp_2(vqa_output)
        return vqa_output, question_type_output

