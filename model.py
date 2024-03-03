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
        self.gru = nn.GRU(word_embedding_size, hidden_size, num_layers= 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1024)
        self.tanh = nn.Tanh()

    def forward(self, input_data):
        output, hidden = self.gru(input_data)
        last_hidden = hidden.squeeze(0)
        embedding = self.fc(last_hidden)
        embedding = self.tanh(embedding)

        return embedding
    
class VQA_header(nn.Module):
    """
    A Visual Question Answering (VQA) header module.
    """

    def __init__(self, ans_vocab_type_dict):
        super().__init__()

        # ModuleDict for VQA headers
        self.vqa_headers = nn.ModuleDict({
            answer_type: nn.Sequential(
                nn.Linear(1024, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, len(ans_vocab_type_dict[answer_type]))
            ) for answer_type in ans_vocab_type_dict.keys()
        })
    def forward(self, hidden_states):
        results = {}
        for question_category in self.vqa_headers.keys():
            outputs = self.vqa_headers[question_category](hidden_states)
            results[question_category] = outputs
        return results
       
class VQA(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_type_dict):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(VQA, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(output_size = args.image_feature_output)

        self.word_embeddings = nn.Embedding(question_vocab_size, args.word_embedding)
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = args.word_embedding,
            hidden_size = args.rnn_hidden_size
            )
        
        self.vqa_mlp = VQA_header(ans_vocab_type_dict)
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
        return output
