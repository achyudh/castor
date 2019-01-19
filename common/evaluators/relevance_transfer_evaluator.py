import torch
import torch.nn.functional as F
import numpy as np

from sklearn import metrics
from .evaluator import Evaluator


class RelevanceTransferEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False):
        super().__init__(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results)
        self.ignore_lengths = False
        self.y_pred = None
        self.docids = None

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        self.y_pred = list()
        self.docid = list()
        total_loss = 0

        # Temp Ave
        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            old_params = self.model.get_params()
            self.model.load_ema_params()

        predicted_labels, target_labels = list(), list()
        for batch_idx, batch in enumerate(self.data_loader):
            if hasattr(self.model, 'TAR') and self.model.TAR:  # TAR condition
                if self.ignore_lengths:
                    scores, rnn_outs = self.model(batch.text)
                else:
                    scores, rnn_outs = self.model(batch.text[0], lengths=batch.text[1])
            else:
                if self.ignore_lengths:
                    scores = self.model(batch.text)
                else:
                    scores = self.model(batch.text[0], lengths=batch.text[1])

            self.docid.extend(batch.docid)
            self.y_pred.extend(scores.cpu().detach().numpy())
            predicted_labels.extend(torch.argmax(scores, dim=1).cpu().detach().numpy())
            target_labels.extend(torch.argmax(batch.label, dim=1).cpu().detach().numpy())
            total_loss += F.cross_entropy(scores, torch.argmax(batch.label, dim=1), size_average=False).item()

            if hasattr(self.model, 'TAR') and self.model.TAR:  # TAR condition
                total_loss += (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        avg_loss = total_loss / len(self.data_loader.dataset.examples)

        # Temp Ave
        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            self.model.load_params(old_params)

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'cross_entropy_loss']
