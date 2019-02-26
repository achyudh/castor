from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from kim_cnn.dropblock import DropBlock1D, LinearScheduler
from lstm_regularization.embed_regularize import embedded_dropout


class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        self.output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.batchnorm = config.batchnorm
        self.beta_ema = config.beta_ema
        self.embed_droprate = config.embed_droprate
        self.dynamic_pool = config.dynamic_pool
        self.dynamic_pool_length = config.dynamic_pool_length
        self.has_bottleneck = config.bottleneck_layer
        self.bottleneck_units = config.bottleneck_units
        self.dropblock_prob = config.dropblock
        self.Ks = 3  # There are three conv nets here

        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
            input_channel = 2
        else:
            print("Unsupported Mode")
            exit()

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (5, words_dim), padding=(4, 0))

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm2 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm3 = nn.BatchNorm2d(self.output_channel)

        if self.dropblock_prob > 0:
            self.dropblock = LinearScheduler(
                DropBlock1D(drop_prob=self.dropblock_prob, block_size=config.dropblock_size),
                start_value=0.,
                stop_value=self.dropblock_prob,
                nr_steps=1000
            )

        self.dropout = nn.Dropout(config.dropout)

        if self.dynamic_pool:
            self.dynamic_pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Dynamic pooling
            if self.has_bottleneck:
                self.fc1 = nn.Linear(self.Ks * self.output_channel * (self.dynamic_pool_length + 1), self.bottleneck_units)
                self.fc2 = nn.Linear(self.bottleneck_units, target_class)
            else:
                self.fc1 = nn.Linear(self.Ks * self.output_channel * (self.dynamic_pool_length + 1), target_class)

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
        if self.mode == 'rand':
            word_input = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(x)
            x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
            x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
            static_input = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)
        else:
            print("Unsupported Mode")
            exit()

        if self.batchnorm:
            x = [F.relu(self.batchnorm1(self.conv1(x))).squeeze(3),
                 F.relu(self.batchnorm2(self.conv2(x))).squeeze(3),
                 F.relu(self.batchnorm3(self.conv3(x))).squeeze(3)]
        else:
            x = [F.relu(self.conv1(x)).squeeze(3),
                 F.relu(self.conv2(x)).squeeze(3),
                 F.relu(self.conv3(x)).squeeze(3)]

        if self.dropblock_prob > 0:
            self.dropblock.step()
            x = [self.dropblock(conv_output) for conv_output in x]

        if self.dynamic_pool:
            x1 = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch, channel_output) * Ks
            x1 = torch.cat(x1, 1)  # (batch, channel_output * Ks)
            x1 = x1.view(-1, self.Ks * self.output_channel * self.dynamic_pool_length)
            x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output, ~=sent_len) * Ks
            x2 = torch.cat(x2, 1)
            x = torch.cat((x1, x2), 1)
        else:
            # (batch, channel_output, ~=sent_len) * Ks
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
            # (batch, channel_output) * Ks
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