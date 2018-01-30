
# coding:utf-8

import numpy
from chainer import cuda, Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as Links
xp = numpy

import argparse
import json
import time
import os
import logging
from collections import deque

import BeamSearch

logging.basicConfig(filename="train.log", level=logging.DEBUG)


class PointerSentinelMixtureModels(Chain):
    def __init__(self, input_size, output_size, feature_size, hidden_size, L, vocab):
        super(PointerSentinelMixtureModels, self).__init__(
            embed=Links.EmbedID(input_size, feature_size),
            H=Links.LSTM(feature_size, hidden_size),
            U=Links.Linear(hidden_size, output_size),
            W=Links.Linear(hidden_size, hidden_size),
            sentinel=Links.Linear(hidden_size, 1, nobias=True)
        )
        self.L = L
        self.vocab = vocab

    def __call__(self, input_data, output_data):
        # encode step
        h, window_words, hidden_states = self.encode_top_state(input_data)
        # decode step
        accum_loss = self.compute_loss("<eos>", output_data[0], window_words, hidden_states)
        for i, word in enumerate(output_data):
            next_word = "<eos>" if (i == len(output_data) - 1) else output_data[i+1]
            accum_loss += self.compute_loss(word, next_word, window_words, hidden_states)
        return accum_loss

    def encode_top_state(self, input_data):
        hidden_states = deque()
        window_words = deque()
        self.H.reset_state()
        for input_vocab in input_data:
            x_i = self.embed(Variable(xp.array([self.vocab[input_vocab] if input_vocab in self.vocab else self.vocab["<unk>"]]).astype(xp.int32)))
            h = self.H(x_i)
            if len(hidden_states) == self.L:
                hidden_states.popleft()
                window_words.popleft()
            hidden_states.append(xp.copy(h.data[0]))
            window_words.append(input_vocab)
        return self.H, window_words, hidden_states

    def decode_one_step(self, input_vocab, window_words, hidden_states):
        x_i = self.embed(Variable(xp.array([self.vocab[input_vocab] if input_vocab in self.vocab else self.vocab["<unk>"]]).astype(xp.int32)))
        h = self.H(x_i)
        q = F.tanh(self.W(h))
        z = [q.data.dot(hidden_state)[0] for hidden_state in hidden_states]
        z = Variable(xp.array([z], dtype=xp.float32))
        z = F.concat((z, self.sentinel(q)), axis=1)
        a = F.softmax(z)
        g = F.select_item(a, xp.array([a.shape[1]-1], dtype=xp.int32))
        rnn_distribution = F.softmax(self.U(h))
        return g, rnn_distribution, a

    def compute_loss(self, input_vocab, output_vocab, window_words, hidden_states):
        g, rnn_distribution, a = self.decode_one_step(input_vocab, window_words, hidden_states)
        # define p_vocab as 0 if output word is not in vocab
        p_vocab = F.select_item(rnn_distribution, xp.array([self.vocab[output_vocab]], dtype=xp.int32)) if output_vocab in self.vocab else Variable(xp.array([0.0], dtype=xp.float32))

        # compute cross entropy
        indexes = [i for i, x in enumerate(window_words) if x==output_vocab]
        exist_var = Variable(xp.array([0], dtype=xp.float32))
        for idx in indexes:
            exist_var += F.select_item(a, xp.array([idx], dtype=xp.int32))
        p_ptr = F.cast(exist_var, xp.float32) if indexes else Variable(xp.array([0.0], dtype=xp.float32))
        cross_entropy = -F.log(F.linear_interpolate(g, p_vocab, p_ptr) + Variable(xp.array([0.01], dtype=xp.float32)))

        # compute attention loss
        attention_loss = F.cast(-F.log(g + exist_var), xp.float32) if indexes else Variable(xp.array([0.0], dtype=xp.float32))
        return cross_entropy + attention_loss

    def reset_state(self):
        self.H.reset_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--embed', '-e', type=int, default=128,
                        help='Number of units in Embed Layer')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of LSTM units')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to save the chainer models')
    parser.add_argument('--vocab_path', help='Specify vocab2id data when you decode')
    parser.add_argument('--id_path', help='Specify id2vocab data when you decode')
    parser.add_argument('-L', type=int, default=30,
                        help='Window size represented as L. Negative value indicates model use all input.')
    parser.add_argument('--mode', help='"train" or "restart" or "decode"')
    parser.add_argument('--model', help='Choose restart model when restart mode')
    parser.add_argument('--beam_size', type=int, default=8, help='Beam size for decoding')
    parser.add_argument('--max_length', type=int, default=50, help='Max length of decode text')
    parser.add_argument('--data_path', help='Specify file path for encode text')
    parser.add_argument('--decode_path', help='Specify file path for saving decode text')
    args = parser.parse_args()

    # model save directory
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # vocab and id2wd save directory
    if not os.path.exists("data"):
        os.mkdir("data")

    embed_dim = args.embed
    hidden_size = args.unit
    window_size = args.L
    mode = args.mode
    vocab_path = args.vocab_path
    id_path = args.id_path

    if mode == "train" or mode == "restart":

        with open("train_data.txt", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]

        if args.mode == "train":
            # make vocab,id2wd
            vocab = {}
            id2wd = {}
            for data in train_data:
                input_data, output_data = data["input"], data["output"]
                for word in input_data:
                    if word not in vocab:
                        id = len(vocab)
                        vocab[word] = id
                        id2wd[id] = word
                for word in output_data:
                    if word not in vocab:
                        id = len(vocab)
                        vocab[word] = id
                        id2wd[id] = word

            # <eos> is a symbol of end of sentence
            id = len(vocab)
            vocab["<eos>"] = id
            id2wd[id] = "<eos>"

            # <unk> is a symbol for the word which never appear in training data
            id = len(vocab)
            vocab["<unk>"] = id
            id2wd[id] = "<unk>"
            vsize = len(vocab)

            with open("data/vocab", "w", encoding="utf-8") as wf:
                wf.write(json.dumps(vocab, ensure_ascii=False))

            with open("data/id2wd", "w", encoding="utf-8") as wf:
                wf.write(json.dumps(id2wd, ensure_ascii=False))

            model = PointerSentinelMixtureModels(vsize, vsize, embed_dim, hidden_size, window_size, vocab)

        if args.mode == "restart":
            with open(vocab_path, "r", encoding="utf-8") as rf:
                vocab = json.loads(rf.read())

            with open(id_path, "r", encoding="utf-8") as rf:
                id2wd = json.loads(rf.read())
            vsize = len(vocab)

            model = PointerSentinelMixtureModels(vsize, vsize, embed_dim, hidden_size, window_size, vocab)
            serializers.load_npz(args.model, model)


        if args.gpu >= 0:
            # GPU setting
            xp = cuda.cupy
            cuda.get_device(args.gpu).use()
            model.to_gpu()

        optimizer = optimizers.Adam()
        optimizer.setup(model)

        n = len(train_data)
        bs = args.batchsize
        early_stopping = False
        epoch = 0
        # repeat reading all training data 10 times
        for i in range(10):
            sffindx = list(numpy.random.permutation(n))
            for j in range(0, n, bs):
                epoch += 1
                s = time.time()
                accum_loss = None
                batch_data = [train_data[idx] for idx in sffindx[j:(j+bs) if (j+bs) < n else n]]
                for data in batch_data:
                    input_data, output_data = data["input"], data["output"]
                    model.reset_state()
                    model.zerograds()
                    # reverse input data for encoder-decoder Models
                    loss = model(reversed([in_word for in_word in input_data]), [out_word for out_word in output_data])
                    accum_loss = loss if accum_loss is None else accum_loss + loss
                logging.info("epoch: {}, batchsize: {}, loss: {}".format(str(epoch), str(bs), str(accum_loss.data[0])))
                accum_loss.backward()
                accum_loss.unchain_backward()
                optimizer.update()
                e = time.time()
                logging.info("epoch {}'s calc time {}".format(str(epoch), str(e-s)))
                # save model every 500 epoch
                if epoch % 500 == 0:
                    outfile = "PointerSentinelMixtureModels-{}.model".format(str(epoch))
                    serializers.save_npz(args.out + "/" + outfile, model)
                    logging.info("Saved Models as {}".format(outfile))

                # early stop if loss of one data less than 0.1
                if accum_loss.data[0] < bs * 0.1:
                    early_stopping = True
                    break

            if early_stopping:
                break

        outfile = "PointerSentinelMixtureModels-Final.model"
        serializers.save_npz(outfile, model)
        logging.info("Saved Models as {}".format(outfile))

    elif mode == "decode":
        beam_size = args.beam_size
        max_length = args.max_length
        data_dir = args.data_path
        decode_dir = args.decode_path

        with open(vocab_path, "r", encoding="utf-8") as rf:
            vocab = json.loads(rf.read())

        with open(id_path, "r", encoding="utf-8") as rf:
            id2wd = json.loads(rf.read())

        with open(data_dir, "r", encoding="utf-8") as rf:
            encode_data = [json.loads(line)["input"] for line in rf]
        vsize = len(vocab)

        model = PointerSentinelMixtureModels(vsize, vsize, embed_dim, hidden_size, window_size, vocab)
        serializers.load_npz(args.model, model)
        model.to_cpu()

        bs = BeamSearch.BeamSearch(model, beam_size, max_length, vocab, id2wd)

        decode_texts = [bs.beam_search(reversed(encode_text))[0].tokens[1:] for encode_text in encode_data]

        with open(decode_dir, "w", encoding="utf-8") as wf:
            for decode_text in decode_texts:
                wf.write("".join(decode_text) + "\n")

if __name__ == "__main__":
    logging.info("Learning Start!")
    main()
