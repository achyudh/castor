import logging
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from datasets.robust04 import Robust04, Robust04Hierarchical, Robust04CharQuantized
from datasets.robust05 import Robust05, Robust05Hierarchical, Robust05CharQuantized
from datasets.robust45 import Robust45, Robust45Hierarchical, Robust45CharQuantized
from han.model import HAN
from kim_cnn.model import KimCNN
from lstm_baseline.model import LSTMBaseline
from lstm_regularization.model import LSTMBaseline as LSTMRegularized
from relevance_transfer.args import get_args
from relevance_transfer.rerank import rerank
from xml_cnn.model import XmlCNN


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, pred_scores, args, topic):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, args.batch_size, args.gpu)
    if args.model == 'HAN':
        saved_model_evaluator.ignore_lengths = True
    dev_acc, dev_precision, dev_ap, dev_f1, dev_loss = saved_model_evaluator.get_scores()[0]

    if split_name == 'test':
        pred_scores[topic] = (saved_model_evaluator.y_pred, saved_model_evaluator.docid)
    else:
        dev_header = 'Dev/Loss Dev/Acc. Dev/Pr. Dev/APr.'
        dev_log_template = '{:4.4f} {:>8.4f}   {:>4.4f} {:4.4f}'
        print('Evaluation metrics for %s split from topic %s' % (split_name, topic))
        print(dev_header)
        print(dev_log_template.format(dev_loss, dev_acc, dev_precision, dev_ap) + '\n')
    return saved_model_evaluator.y_pred


def save_ranks(pred_scores, output_path):
    with open(output_path, 'w') as output_file:
        for topic in pred_scores:
            scores, docid = pred_scores[topic]
            max_scores = defaultdict(list)
            for score, docid in zip(scores, docid):
                max_scores[docid].append(score)

            print("Saving %d results for topic %s..." % (len(max_scores), topic))
            sorted_score = sorted(((sum(scores)/len(scores), docid) for docid, scores in max_scores.items()), reverse=True)
            rank = 1  # Reset rank counter to one
            for score, docid in sorted_score:
                output_file.write(f'{topic} Q0 {docid} {rank} {score} Castor\n')
                rank += 1


if __name__ == '__main__':
    # Load command line args
    args = get_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: You are using CPU for training')
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = get_logger()

    dataset_map = {
        'Robust04': Robust04,
        'Robust45': Robust45,
        'Robust05': Robust05
    }

    dataset_map_hi = {
        'Robust04': Robust04Hierarchical,
        'Robust45': Robust45Hierarchical,
        'Robust05': Robust05Hierarchical
    }

    dataset_map_cq = {
        'Robust04': Robust04CharQuantized,
        'Robust45': Robust45CharQuantized,
        'Robust05': Robust05CharQuantized
    }

    model_map = {
        'LSTMBaseline': LSTMBaseline,
        'LSTMRegularized': LSTMRegularized,
        'KimCNN': KimCNN,
        'HAN': HAN,
        'XML-CNN': XmlCNN
    }

    if args.model == 'HAN':
        dataset = dataset_map_hi[args.dataset]
    else:
        dataset = dataset_map[args.dataset]
    print('Dataset:', args.dataset)

    if args.rerank:
        rerank(args, dataset)

    else:
        topic_iter = 0
        cache_path = os.path.splitext(args.output_path)[0] + '.pkl'
        if args.resume_snapshot:
            # Load previous cached run
            with open(cache_path, 'rb') as cache_file:
                pred_scores = pickle.load(cache_file)
        else:
            pred_scores = dict()

        for topic in dataset.TOPICS:
            topic_iter += 1
            # Skip topics that have already been predicted
            if args.resume_snapshot and topic in pred_scores:
                continue

            print("Training on topic %d of %d..." % (topic_iter, len(dataset.TOPICS)))
            train_iter, dev_iter, test_iter = dataset.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir,
                                                            topic, batch_size=args.batch_size, device=args.gpu,
                                                            unk_init=UnknownWordVecCache.unk)

            config = deepcopy(args)
            config.target_class = 1
            config.dataset = train_iter.dataset
            config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

            print('Vocabulary size:', len(train_iter.dataset.TEXT_FIELD.vocab))
            print('Target Classes:', train_iter.dataset.NUM_CLASSES)
            print('Train Instances:', len(train_iter.dataset))
            print('Dev Instances:', len(dev_iter.dataset))
            print('Test Instances:', len(test_iter.dataset))

            model = model_map[args.model](config)

            if args.cuda:
                model.cuda()
                print('Shifting model to GPU...')

            parameter = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

            if args.dataset not in dataset_map:
                raise ValueError('Unrecognized dataset')
            else:
                train_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, train_iter,
                                                                 args.batch_size, args.gpu)
                test_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, test_iter,
                                                                args.batch_size, args.gpu)
                dev_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, dev_iter,
                                                               args.batch_size, args.gpu)

            trainer_config = {
                'optimizer': optimizer,
                'batch_size': args.batch_size,
                'log_interval': args.log_every,
                'dev_log_interval': args.dev_every,
                'patience': args.patience,
                'model_outfile': args.save_path,
                'logger': logger,
                'resample': args.resample
            }

            if args.model == 'HAN':
                trainer_config['ignore_lengths'] = True
                dev_evaluator.ignore_lengths = True
                test_evaluator.ignore_lengths = True

            trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator,
                                                 test_evaluator, dev_evaluator)

            if not args.trained_model:
                trainer.train(args.epochs)
            else:
                if args.cuda:
                    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
                else:
                    model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

            # Calculate dev and test metrics
            model = torch.load(trainer.snapshot_path)

            if args.model == 'LSTMRegularized':
                if model.beta_ema > 0:
                    old_params = model.get_params()
                    model.load_ema_params()

            if args.dataset not in dataset_map:
                raise ValueError('Unrecognized dataset')
            else:
                evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, pred_scores, args, topic)
                evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, pred_scores, args, topic)

            if args.model == 'LSTMRegularized':
                if model.beta_ema > 0:
                    model.load_params(old_params)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(pred_scores, cache_file)

        save_ranks(pred_scores, args.output_path)
