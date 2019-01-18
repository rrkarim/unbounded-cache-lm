import torch
from torch.autograd import Variable
from cachemodel.model import Cache, CacheKMeans


class Predictor(object):
    def __init__(self, model, src_vocab, tgt_vocab, cache=False, alpha=0):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if cache == True:
            self.cache = CacheKMeans(self.src_vocab, alpha)
        else:
            self.cache = None

    def get_decoder_features(self, src_seq, cache=None):
        src_id_seq = torch.LongTensor(
            [self.src_vocab.stoi[tok] for tok in src_seq]
        ).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            softmax_list, hidden, other = self.model(
                src_id_seq, [len(src_seq)], cache=cache
            )

        return other, hidden

    def predict_with_cache(self, other, hidden):
        alpha = self.cache.smooth
        cache_p = self.cache.calculate_sum(other["hidden"])
        p = alpha * other["sequence_sm"] + (1.0 - alpha) * cache_p
        p = F.log_softmax(p)
        symbols = p.topk(1)[1]

    def predict(self, src_seq, cache):
        other, hidden = self.get_decoder_features(src_seq, cache)

        length = other["length"][0]

        tgt_id_seq = [other["sequence"][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq

    def predict_n(self, src_seq, n=1):
        other, hidden = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other["topk_length"][0][x]
            tgt_id_seq = [
                other["topk_sequence"][di][0, x, 0].data[0] for di in range(length)
            ]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
