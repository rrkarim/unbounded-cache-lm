import random, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .baseRNN import BaseRNN


class DecoderRNN(BaseRNN):
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        sos_id,
        eos_id,
        n_layers=1,
        rnn_cell="gru",
        bidirectional=False,
        input_dropout_p=0,
        dropout_p=0,
        use_attention=False,
    ):
        super(DecoderRNN, self).__init__(
            vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell
        )
        return None
