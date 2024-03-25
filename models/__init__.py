from models.LSTM_VGG import LSTM_VGG
from models.LSTM_VGG_BERT_Hie import LSTM_VGG_BERT_Hie
from models.LSTM_VGG_LSTM_Hie import LSTM_VGG_LSTM_Hie
from models.LSTM_VGG_BiLSTM_Hie import LSTM_VGG_BiLSTM_Hie
from models.LSTM_VGG_BERT_Hie_v1 import LSTM_VGG_BERT_Hie_v1


def call_model(model_name):
    return globals()[model_name]