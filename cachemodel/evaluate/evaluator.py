from __future__ import print_function, division
import cachemodel
from cachemodel.loss import NLLLoss
import torch
import torchtext


class Evaluator(object):
    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):

        model.eval()
        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data,
            batch_size=self.batch_size,
            sort=True,
            sort_key=lambda x: len(x.src),
            device=device,
            train=False,
        )
        tgt_vocab = data.fields[cachemodel.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[cachemodel.tgt_field_name].pad_token]

        with torch.no_grad():  # maybe model.test()?
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(
                    batch, cachemodel.src_field_name
                )
                target_variables = getattr(batch, cachemodel.tgt_field_name)

                decoder_outputs, decoder_hidden, other = model(
                    input_variables, input_lengths.tolist(), target_variables
                )

                seqlist = other["sequence"]  # not sure

                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(
                        step_output.view(target_variables.size(0), -1), target
                    )

                    non_padding = target.ne(pad)
                    correct = (
                        seqlist[step]
                        .view(-1)
                        .eq(target)
                        .masked_select(non_padding)
                        .sum()
                        .item()
                    )
                    match += correct
                    total += non_padding.sum().item()

            if total == 0:
                accuracy = float("nan")
            else:
                accuracy = match / total

            return loss.get_loss(), accuracy
