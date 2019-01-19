import csv
import os
import pickle
import re
import sys

import numpy as np
import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

csv.field_size_limit(sys.maxsize)


def clean_string(string):
    """
    Performs tokenization and string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()[:500]


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')


def char_quantize(string, max_length=1000):
    identity = np.identity(len(Robust45CharQuantized.ALPHABET))
    quantized_string = np.array([identity[Robust45CharQuantized.ALPHABET[char]] for char in list(string.lower()) if char in Robust45CharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(Robust45CharQuantized.ALPHABET)), dtype=np.float32)))


def clean_string_fl(string):
    """
    Returns only the title and first line (excluding the title) for every news article, then calls clean_string
    """
    split_string = string.split('.')
    if len(split_string) > 1:
            return clean_string(split_string[0] + ". " + split_string[1])
    else:
        return clean_string(string)


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return [float(x) for x in string]


def process_docids(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    try:
        docid = int(string)
    except ValueError:
        docid = 0
    return docid


class Robust45(TabularDataset):
    NAME = 'Robust45'
    NUM_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    DOCID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_docids)
    TOPICS = ['307', '310', '321', '325', '330', '336', '341', '344', '345', '347', '350', '353', '354', '355', '356',
              '362', '363', '367', '372', '375', '378', '379', '389', '393', '394', '397', '399', '400', '404', '408',
              '414', '416', '419', '422', '423', '426', '427', '433', '435', '436', '439', '442', '443', '445', '614',
              '620', '626', '646', '677', '690']

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train, validation, test, **kwargs):
        return super(Robust45, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL_FIELD), ('docid', cls.DOCID_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, topic, batch_size=64, shuffle=True, device=0,
              vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param topic: topic from which articles should be fetched
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train_path = os.path.join('TREC', 'data', 'robust45_train_%s.tsv' % topic)
        dev_path = os.path.join('TREC', 'data', 'robust45_dev_%s.tsv' % topic)
        test_path = os.path.join('TREC', 'data', 'core17_%s.tsv' % topic)
        train, val, test = cls.splits(path, train=train_path, validation=dev_path, test=test_path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class Robust45CharQuantized(Robust45):
    ALPHABET = dict(map(lambda t: (t[1], t[0]),
                        enumerate(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    TEXT_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=char_quantize)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, topic, batch_size=64, shuffle=True, device=0,
              vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param topic: topic from which articles should be fetched
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        train_path = os.path.join('TREC', 'data', 'robust45_train_%s.tsv' % topic)
        dev_path = os.path.join('TREC', 'data', 'robust45_dev_%s.tsv' % topic)
        test_path = os.path.join('TREC', 'data', 'core17_%s.tsv' % topic)
        train, val, test = cls.splits(path, train=train_path, validation=dev_path, test=test_path)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)


class Robust45Hierarchical(Robust45):
    IN_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(IN_FIELD, tokenize=split_sents)
