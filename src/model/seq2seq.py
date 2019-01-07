import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, decode_fucntion=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.deocder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(
        self,
        input_variable,
        input_lengths=None,
        target_variable=None,
        teacher_forcing_ratio=0,
    ):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(
            inputs=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

    return result
