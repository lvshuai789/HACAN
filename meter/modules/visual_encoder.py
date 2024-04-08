import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star


class EncoderImagePrecompAttn(nn.Module):
    def __init__(self, img_dim, embed_size, data_name, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        if self.data_name == 'f30k_precomp':
            self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""

        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

        # features = torch.mean(rnn_img,dim=1)
        features = hidden_state[0]

        if self.data_name == 'f30k_precomp':
            features = self.bn(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd


def l2norm(X, dim=1, eps=1e-8):
    """L2-normalize columns of X """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Visual_Enconder(nn.Module):
    def __init__(self, visual_dim, factor, act='leaky_relu'):
        super(Visual_Enconder, self).__init__()
        self.v_fc1 = nn.Linear(visual_dim, visual_dim)
        self.v_drop = nn.Dropout(p=0.2, inplace=False)
        self.v_fc2 = nn.Linear(visual_dim, (visual_dim + factor) // 2)
        self.v_fc3 = nn.Linear((visual_dim + factor) // 2, factor)

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
