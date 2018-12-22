""" Base class for RNN """
import torch.nn as nn

class BaseRNN(nn.Module):
    """
        Class provides the interface for multilayer RNN
        Note:
            This class can not be used directly
        Args:
            TODO
    """
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise NotImplementedError('Unsupported RNN cell type')

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Direct access to the base class is not supported')
