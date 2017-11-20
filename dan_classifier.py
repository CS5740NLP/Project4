"""Simple deep averaging net classifier"""
import os
import pickle
from time import clock

import dynet_config
dynet_config.set(random_seed=42, autobatch=1)

import dynet as dy

MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
VOCAB_SIZE = 4748 # $@$ Task1 result


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


class DANClassifier(object):
    def __init__(self, params, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = params.add_lookup_parameters((vocab_size, hidden_dim))

        self.W_hid = params.add_parameters((hidden_dim, hidden_dim))
        self.b_hid = params.add_parameters((hidden_dim))

        self.w_clf = params.add_parameters((1, hidden_dim))
        self.b_clf = params.add_parameters((1))

    def _predict(self, batch, train=True):

        # load the network parameters
        W_hid = dy.parameter(self.W_hid)
        b_hid = dy.parameter(self.b_hid)
        w_clf = dy.parameter(self.w_clf)
        b_clf = dy.parameter(self.b_clf)

        probas = []
        # predict the probability of positive sentiment for each sentence
        for _, sent in batch:
            sent_embed = [dy.lookup(self.embed, w) for w in sent]
            
            dropout_embed = []
            # $@$ Task3 implying dropout to regluarization training
            if train == True:
                for embed in sent_embed:
                    embed = dy.dropout(embed, 0.5)
                    dropout_embed.append(embed)
                sent_embed = dy.average(dropout_embed)
            else: sent_embed = dy.average(sent_embed)

            # hid = tanh(b + W * sent_embed)
            # but it's faster to use affine_transform in dynet
            hid = dy.affine_transform([b_hid, W_hid, sent_embed])
            hid = dy.tanh(hid)

            y_score = dy.affine_transform([b_clf, w_clf, hid])
            y_proba = dy.logistic(y_score)
            probas.append(y_proba)

        return probas

    def batch_loss(self, sents, train=True):
        probas = self._predict(sents, train)

        # we pack all predicted probas into one vector of length batch_size
        probas = dy.concatenate(probas)

        # we make a dynet vector out of the true ys
        y_true = dy.inputVector([y for y, _ in sents])

        # classification loss: we use the logistic loss
        # this function automatically sums over all entries.
        total_loss = dy.binary_log_loss(probas, y_true)

        return total_loss

    def num_correct(self, sents):
        probas = self._predict(sents, train=False)
        probas = [p.value() for p in probas]
        y_true = [y for y, _ in sents]
        l = len(y_true)

        correct = 0
        # $@$ Task2 job is just here below
        for i in range (0, l):
            if probas[i] > 0.5 and y_true[i] == True:
                correct += 1
            elif probas[i] <= 0.5 and y_true[i] == False:
                correct += 1
            else:
                correct += 0
        # FIXME: count the number of correct predictions here

        return correct


if __name__ == '__main__':

    with open(os.path.join('processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)

    with open(os.path.join('processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    # initialize dynet parameters and learning algorithm
    params = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(params)
    clf = DANClassifier(params, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

    train_batches = make_batches(train_ix, BATCH_SIZE)
    valid_batches = make_batches(valid_ix, BATCH_SIZE)

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        total_loss = 0
        for batch in train_batches:
            dy.renew_cg()
            loss = clf.batch_loss(batch, train=True)
            
            loss.backward()
            trainer.update()
            total_loss += loss.value()

        # iterate over all validation batches, accumulate # correct pred.
        valid_acc = 0
        for batch in valid_batches:
            dy.renew_cg()
            valid_acc += clf.num_correct(batch)

        valid_acc /= len(valid_ix)

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train loss: {:8.5f} "
               "Valid accuracy: {:8.2f}").format(
            it,
            toc - tic,
            total_loss / len(train_ix),
            valid_acc * 100
            ))
