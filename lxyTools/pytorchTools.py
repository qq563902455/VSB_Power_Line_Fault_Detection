import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import math
import os
import random

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

def set_random_seed(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class myBaseModule():
    def __init__(self, random_seed):
        self.random_seed = random_seed
        set_random_seed(self.random_seed)

    def fit(self, x, y, epoch_nums, batch_size, valid_x, valid_y, custom_metric):

        set_random_seed(self.random_seed)

        self.batch_size = batch_size

        x_train = torch.tensor(x, dtype=torch.float32).cuda()
        y_train = torch.tensor(y, dtype=torch.float32).cuda()
        x_val = torch.tensor(valid_x, dtype=torch.float32).cuda()
        y_val = torch.tensor(valid_y, dtype=torch.float32).cuda()

        loss_fn = torch.nn.BCELoss(reduction="sum")

        optimizer = self.optimizer

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch_num: 1/math.sqrt(epoch_num+1))

        train = torch.utils.data.TensorDataset(x_train, y_train)
        valid = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        for epoch in range(epoch_nums):
            scheduler.step()
            print('lr:\t', scheduler.get_lr()[0])

            start_time = time.time()
            self.train()
            avg_loss = 0.
            avg_l2_reg = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):

                y_pred = self(x_batch)
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
                y_pred = y_pred.detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

            score = custom_metric(valid_y, valid_preds)

            elapsed_time = time.time() - start_time

    #         print('epoch ', epoch, 'avg loss:\t', avg_loss, 'max F1:\t', score, 'elapsed_time:\t', int(elapsed_time), 'threshold:\t', threshold)
            print('Epoch {}/{} \t loss={:.4f}  \t l2={:.4f} \t val_loss={:.4f} \t score={:.4f} \t time={:.2f}s'.format(
                epoch + 1, epoch_nums, avg_loss/batch_size, avg_l2_reg, avg_val_loss/batch_size, score, elapsed_time))

    def predict_proba(self, x):

        x_cuda = torch.tensor(x, dtype=torch.float32).cuda()
        test = torch.utils.data.TensorDataset(x_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

        batch_size = self.batch_size

        test_preds = np.zeros(len(x))
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = self(x_batch)
            y_pred = y_pred.detach()
            test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]

        return test_preds
