from copy import deepcopy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from datasets.robust04 import Robust04
from datasets.robust05 import Robust05
from datasets.robust45 import Robust45
from relevance_transfer.args import get_args
from lstm_regularization.model import LSTMBaseline as LSTMRegularized
from lstm_baseline.model import LSTMBaseline
from kim_cnn.model import KimCNN

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


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, pred_scores, batch_size, device, topic):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    scores, metric_names = saved_model_evaluator.get_scores()
    if split_name == 'test':
        pred_scores[topic] = (saved_model_evaluator.y_pred, saved_model_evaluator.docid)
    print('Evaluation metrics for %s split from topic %s' % (topic, split_name))
    print(metric_names)
    print(scores)
    return saved_model_evaluator.y_pred


def save_ranks(pred_scores, output_path, limit=10000):
    with open(output_path, 'w') as output_file:
        for topic in pred_scores:
            scores, docid = pred_scores[topic]
            scores = np.array(scores)[:, 1]
            s_min, s_max = min(scores), max(scores)
            scores = (scores - s_min) / (s_max - s_min)
            sorted_score = sorted(list(zip(docid, scores)), key=lambda x: -x[1])

            rank = 1
            for docid, score in sorted_score:
                output_file.write(f'{topic} Q0 {docid} {rank} {score} Castor\n')
                rank += 1
                if rank > limit:
                    break


if __name__ == '__main__':
    # Set default configuration in : args.py
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

    model_map = {
        'LSTMBaseline': LSTMBaseline,
        'LSTMRegularized': LSTMRegularized,
        'KimCNN': KimCNN
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    dataset = dataset_map[args.dataset]
    pred_scores = dict()

    print('Dataset {} Mode {}'.format(args.dataset, args.mode))
    topic_iter = 0

    for topic in dataset.TOPICS:
        topic_iter += 1
        print("Training on topic %d of %d..." % (topic_iter, len(dataset.TOPICS)))
        train_iter, dev_iter, test_iter = dataset.iters(args.data_dir, args.word_vectors_file, args.word_vectors_dir,
                                                        topic, batch_size=args.batch_size, device=args.gpu,
                                                        unk_init=UnknownWordVecCache.unk)

        config = deepcopy(args)
        config.dataset = train_iter.dataset
        config.target_class = train_iter.dataset.NUM_CLASSES
        config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

        print('Vocabulary size:', len(train_iter.dataset.TEXT_FIELD.vocab))
        print('Target Classes:', train_iter.dataset.NUM_CLASSES)
        print('Train Instances:', len(train_iter.dataset))
        print('Dev Instances:', len(dev_iter.dataset))
        print('Test Instances:', len(test_iter.dataset))

        if args.resume_snapshot:
            if args.cuda:
                model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
            else:
                model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
        else:
            if args.model not in model_map:
                raise ValueError('Unrecognized model')
            else:
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
        }

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
            evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, pred_scores,
                             args.batch_size, args.gpu, topic)
            evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, pred_scores,
                             args.batch_size, args.gpu, topic)

        if args.model == 'LSTMRegularized':
            if model.beta_ema > 0:
                model.load_params(old_params)

    save_ranks(pred_scores, "run.core17.lstm.topics.%s.txt" % args.dataset.lower())
