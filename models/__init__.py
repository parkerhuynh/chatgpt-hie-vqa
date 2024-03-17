from models.LSTM_VGG import LSTM_VGG
from models.LSTM_VGG_BERT_Hie import LSTM_VGG_BERT_Hie
from models.LSTM_VGG_LSTM_Hie import LSTM_VGG_LSTM_Hie


def call_model(model_name):
    return globals()[model_name]