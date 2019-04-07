from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hr_cnn.sentence_encoder import SentenceEncoder


class HRCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_channel = config.output_channel
        self.mode = config.mode
        self.batchnorm = config.batchnorm
        self.beta_ema = config.beta_ema
        self.dynamic_pool = config.dynamic_pool
        self.dynamic_pool_length = config.dynamic_pool_length
        self.has_bottleneck = config.bottleneck_layer
        self.bottleneck_units = config.bottleneck_units
        self.sentence_encoder = SentenceEncoder(config)
        self.Ks = 3

        input_channel = 3
        target_class = config.target_class
        input_channel_dim = config.sentence_channel * self.dynamic_pool_length if self.dynamic_pool \
            else config.sentence_channel

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, input_channel_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (5, input_channel_dim), padding=(4, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (7, input_channel_dim), padding=(6, 0))

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm2 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm3 = nn.BatchNorm2d(self.output_channel)

        self.dropout = nn.Dropout(config.dropout)

        if self.dynamic_pool:
            self.dynamic_pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Dynamic pooling
            if self.has_bottleneck:
                self.fc1 = nn.Linear(self.Ks * self.output_channel * self.dynamic_pool_length, self.bottleneck_units)
                self.fc2 = nn.Linear(self.bottleneck_units, target_class)
            else:
                self.fc1 = nn.Linear(self.Ks * self.output_channel * self.dynamic_pool_length, target_class)

        else:
            if self.has_bottleneck:
                self.fc1 = nn.Linear(self.Ks * self.output_channel, self.bottleneck_units)
                self.fc2 = nn.Linear(self.bottleneck_units, target_class)
            else:
                self.fc1 = nn.Linear(self.Ks * self.output_channel, target_class)

        if self.beta_ema > 0:
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.0

    def forward(self, x, **kwargs):
        x = x.permute(1, 0, 2)  # (sentences, batch size, words)
        num_sentences = x.size()[0]
        x_encoded = list()
        for i in range(num_sentences):
            x_encoded.append(self.sentence_encoder(x[i, :, :]))

        x = torch.stack(x_encoded)  # (sentences, channels, batch size, words)
        x = x.permute(2, 1, 0, 3)  # (batch size, channels, sentences, words)

        if self.batchnorm:
            x = [F.relu(self.batchnorm1(self.conv1(x))).squeeze(3),
                 F.relu(self.batchnorm2(self.conv2(x))).squeeze(3),
                 F.relu(self.batchnorm3(self.conv3(x))).squeeze(3)]
        else:
            x = [F.relu(self.conv1(x)).squeeze(3),
                 F.relu(self.conv2(x)).squeeze(3),
                 F.relu(self.conv3(x)).squeeze(3)]

        if self.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = x.view(-1, self.Ks * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output, ~=sent_len) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)

        x = self.dropout(x)

        if self.has_bottleneck:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logit = self.fc2(x)
        else:
            logit = self.fc1(x)  # (batch, target_size)
        return logit

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params