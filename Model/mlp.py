import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, output_size, activation_f=nn.ReLU, dropout=0.0):
        super(MLP, self).__init__()
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
            nn.Linear(32, output_size),
            )
        

    def forward(self, x):
        out = self.fc(x)
        return out
