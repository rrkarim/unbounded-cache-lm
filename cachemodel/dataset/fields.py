import logging
import torchtext


class SourceField(torchtext.data.Field):
    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        kwargs["batch_first"] = True
        kwargs["include_lengths"] = True

        super(SourceField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):
    SYM_SOS = "<sos>"
    SYM_EOS = "<eos>"

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)
        kwargs["batch_first"] = True

        if kwargs.get("preprocessing") is None:
            kwargs["preprocessing"] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs["preprocessing"]
            kwargs["preprocessing"] = (
                lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]
            )

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
