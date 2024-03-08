from models.LSTM_VGG import LSTM_VGG


def call_model(model_name):
    return globals()[model_name]