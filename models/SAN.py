import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, image_feature_output=1024):
        super(ImageEncoder, self).__init__()
        self.cnn = models.vgg16(pretrained=True).features
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, image_feature_output),
            nn.Tanh())

    def forward(self, image):
        
        image = self.cnn(image)
        image = image.view(-1, 512, 196).transpose(1, 2)
        image_embedding = self.fc(image)
        return image_embedding


class QuestionEncoder(nn.Module):

    def __init__(self, word_embedding=500, rnn_hidden_size=1024):
        super(QuestionEncoder, self).__init__()
        self.lstm = nn.LSTM(word_embedding, rnn_hidden_size, num_layers=2, batch_first=True)
    def forward(self, ques):
        
        _, hx = self.lstm(ques)
        h, _ = hx
        ques_embedding = h[-1]
        return ques_embedding
    
class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = torch.tanh(hi+hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = torch.softmax(ha, dim=1)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u

class SAN(nn.Module):
    # num_attention_layer and num_mlp_layer not implemented yet
    def __init__(self, args,question_vocab_size, ans_vocab_size):          # embed_size, word_embed_size, num_layers, hidden_size
        super(SAN, self).__init__()

        self.img_encoder = ImageEncoder(args.image_feature_output)
        self.word_embeddings = nn.Embedding(question_vocab_size, args.word_embedding)
        self.ques_encoder = QuestionEncoder(
            word_embedding = args.word_embedding, 
            rnn_hidden_size = args.rnn_hidden_size)
        
        self.san = nn.ModuleList([Attention(d=args.image_feature_output, k=args.att_ff_size)] * args.num_att_layers)
        self.mlp = nn.Sequential(
                nn.Linear(args.image_feature_output, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, images, questions):

        image_embeddings = self.img_encoder(images)
        embeds = self.word_embeddings(questions)
        ques_embeddings = self.ques_encoder(embeds)

        vi = image_embeddings
        u = ques_embeddings
        for attn_layer in self.san:
            u = attn_layer(vi, u)
            
        vqa_output = self.mlp(u)
        return vqa_output