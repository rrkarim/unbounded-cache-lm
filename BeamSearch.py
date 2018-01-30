# coding: utf-8
import numpy
from chainer import cuda

class Hypothesis(object):
    """Defines a hypothesis during beam search.
       Reference: TensorFlow Textsum Model beam_search module. https://github.com/tensorflow/models/blob/master/textsum/beam_search.py
    """

    def __init__(self, tokens, log_prob, state):
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def extend(self, token, log_prob, new_state):
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ("Hypothesis(log prob = %.4f, tokens = %s)" % (self.log_prob, self.tokens))


class BeamSearch(object):
    def __init__(self, model, beam_size, max_length, vocab, id2wd):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.vocab = vocab
        self.id2wd = id2wd

    def beam_search(self, input_data):
        h, window_words, hidden_states = self.model.encode_top_state(input_data)
        hyps = [Hypothesis(["<eos>"], 0.0, h)] * self.beam_size
        results = []
        steps = 0

        while steps < self.max_length and len(results) < self.beam_size:
            latest_tokens = [hyp.latest_token for hyp in hyps]
            states = [hyp.state for hyp in hyps]

            all_hyps = []
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            for hyp in hyps:
                topk_tokens, topk_log_probs, new_state = self.decode_topk(hyp.latest_token, hyp.state, window_words, hidden_states)
                for token, log_prob in zip(topk_tokens, topk_log_probs):
                    all_hyps.append(hyp.extend(token, log_prob, new_state))
                if steps == 0:
                    break

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for hyp in self.best_hyps(all_hyps):
                if hyp.latest_token == "<eos>":
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(hyp)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(hyp)
                if len(hyps) == self.beam_size or len(results) == self.beam_size:
                    break

            steps += 1

        if steps == self.max_length:
            results.extend(hyps)

        return self.best_hyps(results)


    def best_hyps(self, hyps):
        return sorted(hyps, key=lambda h:h.log_prob/len(h.tokens), reverse=True)

    def decode_topk(self, token, state, window_words, hidden_states):
        # set LSTM state
        self.model.H = state
        g, rnn_distribution, a = self.model.decode_one_step(token, window_words, hidden_states)
        g_log_pvocab = numpy.log(cuda.to_cpu(rnn_distribution.data[0] * g.data[0]))
        # last element is sentinel vector
        g_log_pptr = numpy.log(cuda.to_cpu(a.data[0] * (1 - g.data[0])))[:-1]
        for idx, word in enumerate(window_words):
            if word in self.vocab:
                g_log_pvocab[self.vocab[word]] += g_log_pptr[idx]
        pvocab_topk = [{"log_prob": g_log_pvocab[i], "word": self.id2wd[str(i)]} for i in numpy.argsort(g_log_pvocab)][:self.beam_size]
        pptr_topk = [{"log_prob": g_log_pptr[i], "word": window_words[i]} for i in numpy.argsort(g_log_pptr)][:self.beam_size if self.beam_size < len(window_words) else len(window_words)]

        pvocab_pptr_topk = pvocab_topk + pptr_topk
        token_topk = []
        log_prob_topk = []
        sorted(pvocab_pptr_topk, key=lambda x:x["log_prob"], reverse=True)

        for word in pvocab_pptr_topk:
            if not word["word"] in token_topk:
                token_topk.append(word["word"])
                log_prob_topk.append(word["log_prob"])

            if len(token_topk) == self.beam_size:
                break
        return token_topk, log_prob_topk, self.model.H
