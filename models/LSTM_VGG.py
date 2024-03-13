import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

    
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
    
class LSTM_VGG(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(LSTM_VGG, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(output_size = args.image_feature_output)

        self.word_embeddings = nn.Embedding(question_vocab_size, args.word_embedding)
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = args.word_embedding,
            hidden_size = args.rnn_hidden_size
            )
        self.mlp = nn.Sequential(
                nn.Linear(args.image_feature_output, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, image, question):
        image = self.image_encoder(image)
        question = self.word_embeddings(question)
        question = self.question_encoder(question)
        combine  = torch.mul(image,question)
        output = self.mlp(combine)
        output = torch.sigmoid(output)
        return output
