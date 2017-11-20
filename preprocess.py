"""
Preprocess all the training data.
Builds a vocabulary and converts all sentences into numerical indices
"""

import os
from glob import glob
from collections import Counter
import pickle
import numpy as np

DATA_DIR = os.environ.get("P4_DATA_ROOT", "data/")


def read_labeled_sentences(f):
    out = []
    for line in f:
        line = line.strip()
        label, sent = line.split("\t")
        words = sent.split()
        bool_label = label == 'POS'
        out.append((bool_label, words))
    return out


def read_unlabeled_sentences(f):
    out = []
    for line in f:
        line = line.strip()
        words = line.split()
        out.append(words)
    return out


if __name__ == '__main__':

    with open(os.path.join(DATA_DIR, "train.txt")) as f:
        train_sents = read_labeled_sentences(f)

    with open(os.path.join(DATA_DIR, "valid.txt")) as f:
        valid_sents = read_labeled_sentences(f)

    with open(os.path.join(DATA_DIR, "test.txt")) as f:
        test_sents = read_labeled_sentences(f)

    with open(os.path.join(DATA_DIR, "unlabeled.txt")) as f:
        unlabeled_sents = read_unlabeled_sentences(f)


    # establish the vocabulary
    ##########################

    # start with list of most common words, labeled or unlabeled
    word_counter = Counter(w for _, sent in train_sents for w in sent)
    word_counter.update(w for sent in unlabeled_sents for w in sent)

    # toss words that don't appear at least 10 times
    vocab = [w for w, _ in word_counter.most_common()
             if word_counter[w] >= 10]

    # add special tokens
    unk = '__UNK__'
    s_start = '__START__'
    s_end = '__END__'
    vocab = [unk, s_start, s_end] + vocab

    # build inverse vocabulary
    # i.e. if vocab[17] = 'cat', then inv_vocab['cat'] = 17
    inv_vocab = {w: k for k, w in enumerate(vocab)}

    def sentence_to_ix(sent):
        sent = [s_start] + sent + [s_end]
        sent_ix = [inv_vocab.get(w, inv_vocab[unk])
                   for w in sent]
        return sent_ix

    # vectorize and save sentence as lists of ids
    #############################################

    train_ix = [(y, sentence_to_ix(sent)) for y, sent in train_sents]
    valid_ix = [(y, sentence_to_ix(sent)) for y, sent in valid_sents]
    test_ix = [(y, sentence_to_ix(sent)) for y, sent in test_sents]
    unlab_ix = [(None, sentence_to_ix(sent)) for sent in unlabeled_sents]

    if not os.path.exists("processed"):
        os.makedirs("processed")

    with open(os.path.join("processed", "vocab.txt"), "w") as f:
        for w in vocab:
            print(w, file=f)

    with open(os.path.join("processed", "train_ix.pkl"), "wb") as f:
        pickle.dump(train_ix, f)

    with open(os.path.join("processed", "unlab_ix.pkl"), "wb") as f:
        pickle.dump(unlab_ix, f)

    with open(os.path.join("processed", "valid_ix.pkl"), "wb") as f:
        pickle.dump(valid_ix, f)

    with open(os.path.join("processed", "test_ix.pkl"), "wb") as f:
        pickle.dump(test_ix, f)
