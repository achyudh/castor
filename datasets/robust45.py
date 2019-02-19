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


def clean_string(string, topic_words=None, max_length=5000):
    """
    Performs tokenization and string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if topic_words is None:
        tokenized_string = string.lower().strip().split()
    else:
        tokenized_string = [x for x in string.lower().strip().split() if x not in topic_words]
    return tokenized_string[:min(max_length, len(tokenized_string))]


def split_sents(string):
    string = re.sub(r"[!?]", " ", string)
    return string.strip().split('.')


def char_quantize(string, max_length=1000):
    identity = np.identity(len(Robust45CharQuantized.ALPHABET))
    quantized_string = np.array([identity[Robust45CharQuantized.ALPHABET[char]] for char in list(string.lower()) if
                                 char in Robust45CharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string,
                               np.zeros((max_length - len(quantized_string), len(Robust45CharQuantized.ALPHABET)),
                                        dtype=np.float32)))


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    # return [float(x) for x in string]
    return 0 if string == '01' else 1


def process_docids(string):
    """
    Returns the docid as an integer
    :param string:
    :return:
    """
    try:
        docid = int(string)
    except ValueError:
        # print("Error converting docid to integer:", string)
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
    TOPIC_WORDS = {'330': {'iran', 'iraq', 'cooperation'}, '427': {'uv', 'damage,', 'eyes'}, '445': {'women', 'clergy'},
                   '336': {'black', 'bear', 'attacks'}, '341': {'airport', 'security'}, '344': {'abuses', 'e', 'mail'},
                   '354': {'journalist', 'risks'}, '355': {'ocean', 'remote', 'sensing'}, '393': {'mercy', 'killing'},
                   '439': {'inventions,', 'scientific', 'discoveries'}, '443': {'u', 's', 'investment,', 'africa'},
                   '345': {'overseas', 'tobacco', 'sales'}, '347': {'wildlife', 'extinction'}, '367': {'piracy'},
                   '307': {'new', 'hydroelectric', 'projects'}, '310': {'radio', 'waves', 'brain', 'cancer'},
                   '321': {'women', 'parliaments'}, '325': {'cult', 'lifestyles'}, '442': {'heroic', 'acts'},
                   '433': {'greek,', 'philosophy,', 'stoicism'}, '435': {'curbing', 'population', 'growth'},
                   '379': {'mainstreaming'}, '408': {'tropical', 'storms'}, '626': {'human', 'stampede'},
                   '423': {'milosevic,', 'mirjana', 'markovic'}, '426': {'law', 'enforcement,', 'dogs'},
                   '350': {'health', 'computer', 'terminals'}, '353': {'antarctica', 'exploration'},
                   '419': {'recycle,', 'automobile', 'tires'}, '422': {'art,', 'stolen,', 'forged'},
                   '356': {'postmenopausal', 'estrogen', 'britain'}, '362': {'human', 'smuggling'},
                   '363': {'transportation', 'tunnel', 'disasters'}, '378': {'euro', 'opposition'},
                   '436': {'railway', 'accidents'}, '690': {'college', 'education', 'advantage'},
                   '414': {'cuba,', 'sugar,', 'exports'}, '416': {'three', 'gorges', 'project'},
                   '614': {'flavr', 'savr', 'tomato'}, '620': {'france', 'nuclear', 'testing'},
                   '389': {'illegal', 'technology', 'transfer'}, '394': {'home', 'schooling'},
                   '400': {'amazon', 'rain', 'forest'}, '404': {'ireland,', 'peace', 'talks'},
                   '646': {'food', 'stamps', 'increase'}, '677': {'leaning', 'tower', 'pisa'},
                   '372': {'native', 'american', 'casino'}, '375': {'hydrogen', 'energy'},
                   '397': {'automobile', 'recalls'}, '399': {'oceanographic', 'vessels'}}

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def make_tokenizer(cls, topic, max_length=5000):
        def tokenizer(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
            string = re.sub(r"\s{2,}", " ", string)
            tokenized_string = [x for x in string.lower().strip().split() if x not in cls.TOPIC_WORDS[topic]]
            return tokenized_string[:min(max_length, len(tokenized_string))]
        return tokenizer

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

        train_path = os.path.join('TREC', 'data', 'robust45_aug_train_%s.tsv' % topic)
        dev_path = os.path.join('TREC', 'data', 'robust45_dev_%s.tsv' % topic)
        test_path = os.path.join('TREC', 'data', 'core17_10k_%s.tsv' % topic)
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
        test_path = os.path.join('TREC', 'data', 'core17_10k_%s.tsv' % topic)
        train, val, test = cls.splits(path, train=train_path, validation=dev_path, test=test_path)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     device=device)


class Robust45Hierarchical(Robust45):
    IN_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(IN_FIELD, tokenize=split_sents)
