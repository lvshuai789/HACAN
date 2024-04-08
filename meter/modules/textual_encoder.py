import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def l2norm(X, dim=1, eps=1e-8):
    """L2-normalize columns of X """

    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Textual_Enconder(nn.Module):
    def __init__(self, textual_dim, factor, act='leaky_relu'):
        super(Textual_Enconder, self).__init__()
        self.v_fc1 = nn.Linear(textual_dim, textual_dim)
        self.v_drop = nn.Dropout(p=0.2, inplace=False)
        self.v_fc2 = nn.Linear(textual_dim, (textual_dim + factor) // 2)
        self.v_fc3 = nn.Linear((textual_dim + factor) // 2, factor)

        self.act = act

    def forward(self, x):
        if self.act == 'leaky_relu':
            x = F.leaky_relu(self.v_fc1(x))
            x = self.v_drop(x)
            x = F.leaky_relu(self.v_fc2(x))
            x = F.leaky_relu(self.v_fc3(x))
            x = l2norm(x, dim=1)

        elif self.act == 'tanh':
            x = F.tanh(self.v_fc1(x))
            x = self.v_drop(x)
            x = F.tanh(self.v_fc2(x))
            x = F.tanh(self.v_fc3(x))
            x = l2norm(x, dim=1)

        else:
            x = self.v_fc1(x)
            x = self.v_drop(x)
            x = self.v_fc2(x)
            x = self.v_fc3(x)
            x = l2norm(x, dim=1)

        return x
