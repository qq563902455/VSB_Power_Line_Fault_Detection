import torch
import torch.nn as nn
import torch.utils.data
import math

import time
import math
import os
import random

from lxyTools.pytorchTools import myBaseModule
from lxyTools.pytorchTools import Attention
from lxyTools.pytorchTools import selfAttention
from lxyTools.pytorchTools import multiHeadAttention
from lxyTools.pytorchTools import set_random_seed
from tqdm import tqdm
import visdom

import numpy as np



class LSTM_softmax(nn.Module, myBaseModule):
    def __init__(self, random_seed,
                       features_dims,
                       seq_len,
                       learning_rate,
                       lstm_out_dim=80,
                       lstm_layers=2,
                       linearReduction_dim=64,
                       env=None):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)

        self.l2_weight = 0.0000
        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.lstm_attention = Attention(lstm_out_dim*2, seq_len)

        self.linear = nn.Linear(lstm_out_dim*4, linearReduction_dim).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.linear2 = nn.Linear(linearReduction_dim, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))
        self.loss_fn = torch.nn.BCELoss(reduction="sum")

        self.vis = env

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm_atten = self.lstm_attention(h_lstm)

        lstm_max_pool, _ = torch.max(h_lstm, 1)

        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)
        nn_out = self.dropout(nn_out)
        nn_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(nn_out))

        return out

class LSTM_selfAttention_softmax(nn.Module, myBaseModule):
    def __init__(self, random_seed,
                       features_dims,
                       seq_len,
                       learning_rate,
                       lstm_out_dim=80,
                       lstm_layers=2,
                       selfAttention_dim=80,
                       linearReduction_dim=64,
                       env=None):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)

        self.l2_weight = 0.0000

        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.selfAttention = selfAttention(selfAttention_dim, selfAttention_dim, 2*lstm_out_dim,dk=64)

        self.lstm_attention = Attention(selfAttention_dim, seq_len)

        self.linear = nn.Linear(2*(selfAttention_dim), linearReduction_dim).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.linear2 = nn.Linear(linearReduction_dim, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))
        self.loss_fn = torch.nn.BCELoss(reduction="sum")

        self.vis = env

    def forward(self, x):

        h_lstm, _ = self.lstm(x)
        h_lstm = self.selfAttention(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        lstm_max_pool, _ = torch.max(h_lstm, 1)
        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)


        nn_out = self.dropout(nn_out)
        nn_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(nn_out))
        return out

class semi_superviesd_LSTM_selfAttention(nn.Module, myBaseModule):
    def __init__(self, random_seed,
                       features_dims,
                       seq_len,
                       learning_rate,
                       unsupervised_out_dim,
                       unsupervised_lr,
                       lstm_out_dim=80,
                       gru_out_dim=80,
                       lstm_layers=2,
                       gru_layers=2,
                       selfAttention_dim=80,
                       selfAttentionSeq_dim=80,
                       linearReduction_dim=64,
                       env=None):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)

        self.l2_weight = 0.0000
        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.gru = nn.GRU(features_dims, gru_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.bn_lstm = nn.LayerNorm([seq_len, lstm_out_dim*4])

        self.selu = nn.SELU()

        self.selfAttention = selfAttention(selfAttention_dim, selfAttention_dim, 2*(lstm_out_dim+gru_out_dim),dk=64)

        self.lstm_attention = Attention(selfAttention_dim+selfAttentionSeq_dim, seq_len)

        self.bn_final = nn.LayerNorm([2*(selfAttention_dim+selfAttentionSeq_dim)])

        self.linear = nn.Linear(2*(selfAttention_dim+selfAttentionSeq_dim), linearReduction_dim).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.linear2 = nn.Linear(linearReduction_dim, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.unsupervised_linear = nn.Linear(2*(selfAttention_dim+selfAttentionSeq_dim), unsupervised_out_dim).cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.unsupervised_optimizer = torch.optim.Adam(self.parameters(), lr=unsupervised_lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))

        self.unsupervised_loss_fn = torch.nn.MSELoss(reduction="sum")
        self.supervised_loss_fn = torch.nn.BCELoss(reduction="sum")

        self.vis = env

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_gru, _ = self.gru(x)

        h_lstm = torch.cat([h_gru, h_lstm], 2)

        # h_lstm = self.selu(h_lstm)
        # h_lstm = self.bn_lstm(h_lstm)
        h_lstm = self.selfAttention(h_lstm)


        h_lstm_atten = self.lstm_attention(h_lstm)
        lstm_max_pool, _ = torch.max(h_lstm, 1)
        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)
        # nn_out = self.selu(nn_out)
        # nn_out = self.bn_final(nn_out)

        nn_out = self.dropout(nn_out)
        unsupervised_out = self.sigmoid(self.unsupervised_linear(nn_out))

        supervised_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(supervised_out))

        return out, unsupervised_out

    def fit(self, x, y, epoch_nums, batch_size, valid_x, valid_y, custom_metric=None, plot_fold=None, unsupervised_flag=False):

        if self.vis is not None:
            vis = visdom.Visdom(env=self.vis)

        set_random_seed(self.random_seed)

        self.batch_size = batch_size

        x_train = torch.tensor(x, dtype=torch.float32).cuda()
        y_train = torch.tensor(y, dtype=torch.float32).cuda()
        x_val = torch.tensor(valid_x, dtype=torch.float32).cuda()
        y_val = torch.tensor(valid_y, dtype=torch.float32).cuda()

        loss_fn = self.supervised_loss_fn
        optimizer = self.optimizer

        if unsupervised_flag:
            loss_fn = self.unsupervised_loss_fn
            optimizer = self.unsupervised_optimizer

        scheduler = self.scheduler

        train = torch.utils.data.TensorDataset(x_train, y_train)
        valid = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        for epoch in range(epoch_nums):
            scheduler.step()
            # print('lr:\t', scheduler.get_lr()[0])

            start_time = time.time()
            self.train()
            avg_loss = 0.
            avg_l2_reg = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):

                y_pred = self(x_batch)
                if unsupervised_flag:
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]

                bceloss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                bceloss.backward()
                optimizer.step()

                avg_loss += bceloss.item() / len(train_loader)

            self.eval()
            valid_preds = np.zeros((x_val.size(0)))

            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = self(x_batch)
                if unsupervised_flag:
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                y_pred = y_pred.detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

            if custom_metric is not None:
                score = custom_metric(valid_y, valid_preds)

            elapsed_time = time.time() - start_time

            if self.vis is not None:
                vis.line(X=torch.Tensor([[epoch, epoch]]),
                         Y=torch.Tensor([[avg_loss/batch_size, avg_val_loss/batch_size]]),
                         win='loss'+plot_fold,
                         opts={'legend':['local_loss', 'valid_loss'],
                               'xlabel': 'epoch',
                               'title': 'train'+plot_fold},
                         update='append' if epoch > 0 else None)

            if custom_metric is not None:
                if self.vis is not None:
                    vis.line(X=torch.Tensor([epoch]),
                             Y=torch.Tensor([score]),
                             win='score'+plot_fold,
                             opts={'legend':['score'],
                                   'xlabel': 'epoch',
                                   'title': 'valid'+plot_fold},
                             update='append' if epoch > 0 else None)

            if custom_metric is not None:
                print('Epoch {}/{} \t loss={:.4f}  \t l2={:.4f} \t val_loss={:.4f} \t score={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epoch_nums, avg_loss/batch_size, avg_l2_reg, avg_val_loss/batch_size, score, elapsed_time))
            else:
                print('Epoch {}/{} \t loss={:.4f}  \t l2={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epoch_nums, avg_loss/batch_size, avg_l2_reg, avg_val_loss/batch_size, elapsed_time))


    def unsupervisedTraining(self, x, y, epoch_nums, batch_size, train_ratio, custom_metric=None, plot_fold=None):

        set_random_seed(self.random_seed)

        samples_num = x.shape[0]
        index_list = np.random.permutation(np.arange(samples_num))

        train_index = index_list[0: int(train_ratio*samples_num)]
        valid_index = index_list[int(train_ratio*samples_num): ]

        train_x = x[train_index]
        train_y = y[train_index]

        valid_x = x[valid_index]
        valid_y = y[valid_index]

        self.fit(train_x, train_y, epoch_nums, batch_size, valid_x, valid_y, custom_metric, plot_fold, unsupervised_flag=True)





    def predict_proba(self, x):

        x_cuda = torch.tensor(x, dtype=torch.float32).cuda()
        test = torch.utils.data.TensorDataset(x_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

        batch_size = self.batch_size

        test_preds = np.zeros(len(x))
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = self(x_batch)[0]
            y_pred = y_pred.detach()
            test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

        return test_preds
