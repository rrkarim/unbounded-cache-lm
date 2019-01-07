import logging
import torchtext


class SourceField(torchtext.data.Field):
    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)
        if kwargs.get("batch_first") is False:
            logger.warning("Option batch_first has to be set to use the model")
            kwargs["batch_first"] = True
        if kwargs.get("include_lengths") is False:
            logger.warning("Option include_lengths has to be set ot use the model")
            kwargs["include_lengths"] = True
        super(SourceField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):
    SYM_SOS = "<sos>"
    SYM_EOS = "<eos>"

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get("batch_first") == False:
            logger.warning(
                "Option batch_first has to be set to use pytorch-seq2seq.  Changed to True."
            )
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

    def build(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab_stoi(self.SYM_SOS)
        self.eos_id = self.vocab_stoi(self.SYM_EOS)
