import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss


class BaseCache(nn.Module):
    def __init__(self, vocab, alpha):
        super(BaseCache, self).__init__()
        self.vectors = [] # high memory usage here
        self.vocab = vocab
        self.word_to_index = {}
        self.kernel = torch.exp
        self.alpha = alpha
        self.cache_elems = []

    def find_smooth(self):
        return self.smooth


class Cache(BaseCache):
    def __init__(self, vocab=None, alpha=0.5, smooth=0.2, device=None):
        super(Cache, self).__init__(vocab, alpha)
        self.smooth = smooth
        self.device = device

    def _add_element(self, item, hidden):
        self.cache_elems.append((item, torch.squeeze(hidden)))
        self.vectors.append(hidden)
        self.word_to_index[item] = len(self.cache_elems) - 1

    def get_sum(self, h_t, v):
        sum_ = 0.0
        for w_i, h_i in self.cache_elems:
            if w_i == v:
                sum_ += self.kernel((torch.norm(h_i - h_t)) / self.smooth)
        return sum_

    def calculate_sum(self, h_t):
        cache_p = torch.zeros([1, len(self.vocab.itos)])
        for i, v in enumerate(self.vocab.itos):
            cache_p[0][i] = self.get_sum(h_t, v)
        cache_p = F.log_softmax(cache_p).to(self.device)
        return cache_p


class CacheKMeans(BaseCache):

    def __init__(self, vocab=None, alpha=0.5, smooth=0.2, kn=10):
        super(CacheKMeans, self).__init__(vocab=vocab, alpha=alpha)
        self.kn = kn

    def get_sum(self, h_t, v):
        sum_ = 0.
        list_indeces, kth_neighb = self._find_neighbors(k=3, h_t=h_t)
        for index in list_indeces:
            w_i, h_i = self.cache_elems[index]
            if w_i == v:
                sum_ += self.kernel((torch.norm(h_i - h_t)) / kth_neighb)


    def _find_neighbors(self, k=1, h_t=None):
        d = h.shape[0]
        quantizer = faiss.IndexFlatL2(d)
        nlist = 100
        m = 8
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        index.train(self.vectors)
        index.add(self.vectors)

        D,I = index.search([h_t], k)
        return I, D[-1][-1]



