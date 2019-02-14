import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = "attention_score"
    KEY_LENGTH = "length"
    KEY_SEQUENCE = "sequence"
    KEY_SEQUENCE_SM = "sequence_sm"
    KEY_HIDDEN = "hidden"

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
            vocab_size,
            max_len,
            hidden_size,
            input_dropout_p,
            dropout_p,
            n_layers,
            rnn_cell,
        )

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(
            hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p
        )

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(
            self.out(output.contiguous().view(-1, self.hidden_size)), dim=1
        ).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(
        self,
        inputs=None,
        encoder_hidden=None,
        encoder_outputs=None,
        function=F.log_softmax,
        teacher_forcing_ratio=0,
        cache=False,
    ):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(
            inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        )
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        sequence_softmax = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn, decoder_hidden):
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if cache is not None:
                df = cache.calculate_sum(torch.squeeze(decoder_hidden))
                with torch.no_grad():
                    average_p = cache.alpha * step_output
                    average_p += (1.0 - cache.alpha) * cache.calculate_sum(torch.squeeze(decoder_hidden))
                average_p = F.log_softmax(average_p)
                decoder_outputs.append(average_p)
            else:
                decoder_outputs.append(step_output)

            symbols = decoder_outputs[-1].topk(1)[1]  # TODO: hardcode
            if cache is not None:
                cache._add_element(symbols.numpy()[0][0], decoder_hidden)

            sequence_symbols.append(symbols)
            sequence_softmax.append(step_output)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:  #
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, function=function
            )

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn, decoder_hidden)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs, function=function
                )
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn, decoder_hidden)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_SEQUENCE_SM] = sequence_softmax
        ret_dict[DecoderRNN.KEY_HIDDEN] = decoder_hidden
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return h

    def _validate_args(
        self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
    ):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError(
                    "Argument encoder_outputs cannot be None when attention is used."
                )

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided."
                )
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
