from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import cachemodel
from cachemodel.evaluator import Evaluator
from cachemodel.loss import NLLLoss
from cachemodel.optim import Optimizer
from cachemodel.util.checkpoint import Checkpoint


class SupervisedTrainer(object):
    def __init__(
        self,
        expt_dir="experiment",
        loss=NLLLoss(),
        batch_size=64,
        random_seed=None,
        checkpoint_every=500,
        print_every=1000,
    ):
        self._trainer = "Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(
        self,
        input_variable,
        input_lengths,
        target_variable,
        model,
        teacher_forcing_ratio,
    ):
        loss = self.loss
        decoder_outputs, decoder_hidden, other = model(
            input_variable,
            input_lengths,
            target_variable,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(
                step_output.contiguous().view(batch_size, -1),
                target_variable[:, step + 1],
            )

        model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.get_loss()

    def _train_epoches(
        self,
        data,
        model,
        n_epochs,
        start_epoch,
        start_step,
        dev_data=None,
        teacher_forcing_ratio=0,
    ):
        log = self.logger

        print_loss_total = 0
        epoch_loss_total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data,
            batch_size=self.batch_size,
            sort=False,
            sort_withing_batch=True,
            sort_key=lambda x: len(x.src),
            device=device,
            repeat=False,
        )

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
