"""Simplest possible neural language model:
    use word w_i to predict word w_(i + 1)
"""
import os
import pickle
from time import clock
from math import exp
import random

import dynet_config
dynet_config.set(random_seed=42, autobatch=1)

import dynet as dy

MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
USE_UNLABELED = True
VOCAB_SIZE = 4748


def make_batches(data, batch_size):
    batches = []
    batch = []
    for pair in data:
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []

        batch.append(pair)

    if batch:
        batches.append(batch)

    return batches


class SimpleNLM(object):

    def __init__(self, params, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embed = params.add_lookup_parameters((vocab_size, hidden_dim))

        self.W_hid = params.add_parameters((hidden_dim, 2*hidden_dim))
        self.b_hid = params.add_parameters((hidden_dim))

        self.W_out = params.add_parameters((vocab_size, hidden_dim))

    def batch_loss(self, batch, train=True):

        # load the parameters
        W_hid = dy.parameter(self.W_hid)
        b_hid = dy.parameter(self.b_hid)

        W_out = dy.parameter(self.W_out)

        losses = []
        for _, sent in batch:
            for i in range(1, len(sent)):
                # task 6 3gram
                if i==1:
                    prev_word2_ix = sent[len(sent)-1]
                else: prev_word2_ix = sent[i-2]
                prev_word_ix = sent[i - 1]
                curr_word_ix = sent[i]

                ctx1 = dy.lookup(self.embed, prev_word_ix)
                ctx2 = dy.lookup(self.embed, prev_word2_ix)
                ctx = dy.concatenate([ctx2,ctx1])

                # hid is the hidden layer output, size=hidden_size
                # compute b_hid + W_hid * ctx, but faster
                hid = dy.affine_transform([b_hid, W_hid, ctx])
                hid = dy.tanh(hid)

                # out is the prediction of the next word, size=vocab_size
                out = W_out * hid

                # Intepretation: The model estimates that
                # log P(curr_word=k | prev_word) ~ out[k]
                # in other words,
                # P(curr_word=k | prev_word) = exp(out[k]) / sum_j exp(out[j])
                #                            = softmax(out)[k]

                # We want to maximize the probability of the correct word.
                # (equivalently, minimize the negative log-probability)

                loss = dy.pickneglogsoftmax(out, curr_word_ix)
                losses.append(loss)

        # esum simply adds up the expressions in the list
        return dy.esum(losses)


if __name__ == '__main__':

    with open(os.path.join('processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)


    # $@$ Task5 building a new training dataset with labeled and 
    # unlabeled dataset together
    if USE_UNLABELED:
        with open(os.path.join('processed', 'unlab_ix.pkl'), 'rb') as f:
            unlab_ix = pickle.load(f)
        train_ix = train_ix + unlab_ix
        random.shuffle(train_ix)

    with open(os.path.join('processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    
   # initialize dynet parameters and learning algorithm
    params = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(params)
    lm = SimpleNLM(params, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

    train_batches = make_batches(train_ix, batch_size=BATCH_SIZE)
    valid_batches = make_batches(valid_ix, batch_size=BATCH_SIZE)

    n_train_words = sum(len(sent) for _, sent in train_ix)
    n_valid_words = sum(len(sent) for _, sent in valid_ix)
    

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        total_loss = 0
        for batch in train_batches:
            dy.renew_cg()
            loss = lm.batch_loss(batch, train=True)
            loss.backward()
            trainer.update()
            total_loss += loss.value()

        # iterate over all validation batches, accumulate loss.
        valid_loss = 0
        for batch in valid_batches:
            dy.renew_cg()
            loss = lm.batch_loss(batch, train=False)
            valid_loss += loss.value()

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train perplexity: {:8.3f} "
               "Valid perplexity: {:8.3f}").format(
            it,
            toc - tic,
            exp(total_loss / n_train_words),
            exp(valid_loss / n_valid_words)
            ))

    # FIXME: make sure to update filenames when implementing ngram models
    fn = "embeds_baseline_lm_3gram"
    if USE_UNLABELED:
        fn += "_unlabeled"

    print("Saving embeddings to {}".format(fn))
    lm.embed.save(fn, "/embed")
