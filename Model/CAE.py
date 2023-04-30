import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

mse_loss = nn.MSELoss()


class Encoder(nn.Module):
    def __init__(self, AE_input_size, AE_output_size = 28, activation_f=nn.ReLU, dropout=0.0):
        super(Encoder, self).__init__()
        print('using deep encoder')
        # TODO input shape hardcoded, change it
        self.encoder = nn.Sequential(nn.Linear(AE_input_size, 512),
                                     activation_f(),
                                     nn.Linear(512, 256),
                                     activation_f(),
                                     nn.Linear(256, 128),
                                     activation_f(),
                                     nn.Linear(128, AE_output_size))

    def forward(self, x):
        x = self.encoder(x)
        return x
