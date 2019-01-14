import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import cachemodel  # TODO
from cachemodel.train import SupervisedTrainer
from cachemodel.model import EncoderRNN, DecoderRNN, Seq2seq, Cache
from cachemodel.loss import Perplexity
from cachemodel.optim import Optimizer
from cachemodel.dataset import SourceField, TargetField
from cachemodel.evaluate import Predictor
from cachemodel.utils.checkpoint import Checkpoint

try:
    raw_input
except NameError:
    raw_input = input


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path", action="store", dest="train_path", help="Path to train data"
)
parser.add_argument(
    "--dev_path", action="store", dest="dev_path", help="Path to dev data"
)
parser.add_argument(
    "--expt_dir",
    action="store",
    dest="expt_dir",
    default="./experiment",
    help="Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided",
)
parser.add_argument(
    "--load_checkpoint",
    action="store",
    dest="load_checkpoint",
    help="The name of the checkpoint to load, usually an encoded time string",
)
parser.add_argument(
    "--resume",
    action="store_true",
    dest="resume",
    default=False,
    help="Indicates if training has to be resumed from the latest checkpoint",
)
parser.add_argument(
    "--log-level", dest="log_level", default="info", help="Logging level."
)

opt = parser.parse_args()

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info(
        "loading checkpoint from {}".format(
            os.path.join(
                opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
            )
        )
    )
    checkpoint_path = os.path.join(
        opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
    )
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    train = torchtext.data.TabularDataset(
        path=opt.train_path,
        format="tsv",
        fields=[("src", src), ("tgt", tgt)],
        filter_pred=len_filter,
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path,
        format="tsv",
        fields=[("src", src), ("tgt", tgt)],
        filter_pred=len_filter,
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = True
        encoder = EncoderRNN(
            len(src.vocab),
            max_len,
            hidden_size,
            bidirectional=bidirectional,
            variable_lengths=True,
        )
        decoder = DecoderRNN(
            len(tgt.vocab),
            max_len,
            hidden_size * 2 if bidirectional else hidden_size,
            dropout_p=0.2,
            use_attention=True,
            bidirectional=bidirectional,
            eos_id=tgt.eos_id,
            sos_id=tgt.sos_id,
        )
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    t = SupervisedTrainer(
        loss=loss,
        batch_size=32,
        checkpoint_every=50,
        print_every=10,
        expt_dir=opt.expt_dir,
    )

    seq2seq = t.train(
        seq2seq,
        train,
        num_epochs=0,
        dev_data=dev,
        optimizer=optimizer,
        teacher_forcing_ratio=0.5,
        resume=opt.resume,
    )

predictor = Predictor(seq2seq, input_vocab, output_vocab, cache=True)
cache = Cache(output_vocab)
seq = []

while True:
    seq_str = raw_input("Type in a source sequence:")
    elem = seq_str.strip()
    seq.append(elem) # we should also remove elements after some threshold
    print(predictor.predict(seq, cache))
