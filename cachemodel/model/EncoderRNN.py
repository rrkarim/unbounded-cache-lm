import torch.nn as nn
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        input_dropout_p=0,
        dropout_p=0,
        n_layers=1,
        bidirectional=False,
        rnn_cell="gru",
        variable_lengths=False,
        embedding=None,
        update_embedding=True,
    ):
        super(EncoderRNN, self).__init__(
            vocab_size,
            max_len,
            hidden_size,
            input_dropout_p,
            dropout_p,
            n_layers,
            rnn_cell,
        )

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p,
        )

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=True
            )
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
