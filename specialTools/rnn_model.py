import torch
import torch.nn as nn
import torch.utils.data

from lxyTools.pytorchTools import myBaseModule
from lxyTools.pytorchTools import Attention

class LSTM_softmax(nn.Module, myBaseModule):
    def __init__(self, random_seed, features_dims, seq_len, learning_rate, lstm_out_dim=80, lstm_layers=2):
        nn.Module.__init__(self)
        myBaseModule.__init__(self, random_seed)
        # lstm_out_dim = 80
        # lstm_layers = 2
        self.l2_weight = 0.0000
        self.lstm = nn.LSTM(features_dims, lstm_out_dim, lstm_layers, bidirectional=True, batch_first=True).cuda()
        self.lstm_attention = Attention(lstm_out_dim*2, seq_len)
        self.linear = nn.Linear(lstm_out_dim*4, 64).cuda()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(64, 1).cuda()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm_atten = self.lstm_attention(h_lstm)
        lstm_max_pool, _ = torch.max(h_lstm, 1)
        nn_out = torch.cat([lstm_max_pool, h_lstm_atten], 1)
        nn_out = self.dropout(nn_out)
        nn_out = self.relu(self.linear(nn_out))
        out = self.sigmoid(self.linear2(nn_out))
        return out
