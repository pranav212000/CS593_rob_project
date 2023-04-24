import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class MLPComplete(nn.Module):
    def __init__(self, input_size, output_size, activation_f=nn.ReLU, dropout=0.5):
        super(MLPComplete, self).__init__()
        self.mse = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            activation_f(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            activation_f(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            activation_f(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            activation_f(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation_f(),
            nn.Dropout(dropout),
            # nn.Linear(1024, 32),
            # nn.BatchNorm1d(32),
            # activation_f(),
            # nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            activation_f(),
            nn.Linear(32, output_size))

        self.opt = None

    def forward(self, x):
        out = self.fc(x)
        return out

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.fc.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.fc.parameters()),
                           lr=lr, momentum=momentum)

    def loss(self, pred, truth):
        return self.mse(pred, truth)

    def step(self, x, y):
        # given a batch of data, optimize the parameters by one gradient descent step
        # assume here x and y are torch tensors, and have been
        self.zero_grad()
        f = self.forward(x)
        loss = self.loss(self.forward(x), y)
        loss.backward()
        self.opt.step()
