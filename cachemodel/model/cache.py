import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseCache(nn.Module):
    def __init__(self):
        super(BaseCache, self).__init__()


class Cache(BaseCache):
    def __init__(self, vocab=None, alpha=0.5, smooth=0.2):
        super(Cache, self).__init__()
        self.cache_elems = []
        self.alpha = alpha
        self.smooth = smooth
        self.vocab = vocab
        self.kernel = np.exp

    def _add_element(self, item, hidden):
        self.cache_elems.append((item, hidden))

    def get_sum(self, h_t, v):
        sum_ = 0.0
        for w_i, h_i in self.cache_elems:
            if w_i == v:
                sum_ += self.kernel((np.linalg.norm(h_i - h_t)) / self.smooth)
        return sum_

    def calculate_sum(self, h_t):
        cache_p = torch.zeros([1, len(self.vocab.itos)])
        for i, v in enumerate(self.vocab.itos):
            cache_p[0][i] = self.get_sum(h_t[i], v)
        cache_p = F.log_softmax(cache_p)
        return cache_p


class CacheKmeans(BaseCache):
    def __init__(self):
        super(CacheKmeans, self).__init__()
