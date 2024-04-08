import torch
import torch.nn as nn
from torch.nn.functional import normalize
# from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention, LayerNorm, Dropout


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, nhead, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout,
                                                 batch_first=False)
        self.norm = LayerNorm(hidden_size)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=81):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :]
        return self.dropout(x)


class FuseTransEncoder(nn.Module):
    def __init__(self, batch_size, num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.self_attn_layer = SelfAttentionLayer(hidden_size, nhead)
        self.batch_size = batch_size
        self.d_model = hidden_size
        # self.sigal_d = int(self.d_model / 2)
        self.sigal_d = 49
        self.pos_encoder = PositionalEncoding(hidden_size)

    # def forward(self, tokens):
    #
    #     # tokens = tokens.reshape(self.batch_size * 81, -1)
    #     tokens = normalize(tokens, p=2, dim=1)
    #     # tokens = tokens.reshape(self.batch_size, 81, 768)
    #     tokens = tokens.transpose(0, 1)  # 调整为 (sequence_length, batch_size, feature_size)
    #     pos_enc = self.pos_encoder(tokens)
    #     tokens = tokens + pos_enc
    #     tokens = self.self_attn_layer(tokens)
    #     # encoder_X_r = self.transformerEncoder(tokens)
    #     encoder_X_r = tokens.transpose(0, 1)  # 调整回 (batch_size, sequence_length, feature_size)
    #
    #
    #
    #     # encoder_X_r = self.transformerEncoder(tokens)
    #     # encoder_X_r = self.self_attn_layer(encoder_X_r)
    #
    #
    #     img, txt = encoder_X_r[:, :self.sigal_d, :], encoder_X_r[:, self.sigal_d:, :]
    #
    #     # img = normalize(img, p=2, dim=1)
    #     # txt = normalize(txt, p=2, dim=1)
    #
    #     # print("img", img.shape)
    #     # print("txt", txt.shape)
    #     # img = img.reshape(self.batch_size, -1)
    #     # txt = txt.reshape(self.batch_size, -1)
    #     return img, txt

    def forward(self, tokens):
        # tokens = tokens.reshape(self.batch_size * 81, -1)
        img, txt = tokens[:, :self.sigal_d, :], tokens[:, self.sigal_d:, :]
        img = img.transpose(0, 1)
        txt = txt.transpose(0, 1)
        img_pos_enc = self.pos_encoder(img)
        txt_pos_enc = self.pos_encoder(txt)

        img = img + img_pos_enc
        txt = txt + txt_pos_enc

        # tokens = torch.cat((img,txt), dim=1)
        # tokens = normalize(tokens, p=2, dim=1)
        # tokens = tokens.reshape(self.batch_size, 81, 768)
        # tokens = tokens.transpose(0, 1)  # 调整为 (sequence_length, batch_size, feature_size)
        # pos_enc = self.pos_encoder(tokens)
        # tokens = tokens + pos_enc
        img = self.self_attn_layer(img)
        txt = self.self_attn_layer(txt)
        # encoder_X_r = self.transformerEncoder(tokens)
        # encoder_X_r = tokens.transpose(0, 1)  # 调整回 (batch_size, sequence_length, feature_size)

        # encoder_X_r = self.transformerEncoder(tokens)
        # encoder_X_r = self.self_attn_layer(encoder_X_r)

        # img, txt = encoder_X_r[:, :self.sigal_d, :], encoder_X_r[:, self.sigal_d:, :]

        # img = normalize(img, p=2, dim=1)
        # txt = normalize(txt, p=2, dim=1)

        # print("img", img.shape)
        # print("txt", txt.shape)
        # img = img.reshape(self.batch_size, -1)
        # txt = txt.reshape(self.batch_size, -1)
        img = img.transpose(0, 1)
        txt = txt.transpose(0, 1)
        return img, txt


def get_activate_func(act_func=None):
    if act_func is None or act_func.lower() == 'id':
        return nn.Identity()
    if act_func.lower() == 'relu':
        return nn.ReLU()
    if act_func.lower() == 'swish':
        pass
        # return MemoryEfficientSwish()
    if act_func.lower() == 'tanh':
        return nn.Tanh()
    if act_func.lower() == 'gelu':
        return nn.GELU()
    if act_func.lower() == 'elu':
        return nn.ELU()


class SeqLinear(nn.Module):
    def __init__(self, ft_in, ft_out=[128], dropout=0.5, batch_norm=True, act_func='relu'):
        super(SeqLinear, self).__init__()
        self.linear = []
        self.norm = []
        self.dropout = []
        self.act = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(nn.Linear(ft_in, ft_out[idx]))
            else:
                self.linear.append(nn.Linear(ft_out[idx - 1], ft_out[idx]))
            if idx != len(ft_out) - 1:
                if batch_norm:
                    # self.norm.append(nn.BatchNorm1d(ft_out[idx]))
                    self.norm.append(nn.LayerNorm([ft_out[idx]]))
                else:
                    self.norm.append(nn.Identity())
                self.act.append(get_activate_func(act_func))
            self.dropout.append(nn.Dropout(p=dropout))

        self.linear = nn.ModuleList(self.linear)
        for x in self.linear:
            nn.init.kaiming_normal_(x.weight)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)

    def forward(self, x):
        # x.shape (in_channel, ft_in)
        for idx in range(len(self.linear)):
            # x = self.linear[idx](x)
            # if idx != (len(self.linear)-1): # last layer not use relu
            #     x = self.norm[idx](x)
            #     x = self.act[idx](x)
            # x = self.dropout[idx](x)
            # C10
            x = self.dropout[idx](x)
            x = self.linear[idx](x)
            if idx != (len(self.linear) - 1):  # last layer not use relu
                x = self.act[idx](x)
                x = self.norm[idx](x)
        return x


def concat_node(x1, x2, n_x1, n_x2):
    x_concat = torch.tensor(()).to(x1.device)
    count1 = 0
    count2 = 0
    for idx in range(len(n_x1)):
        x_concat = torch.cat((x_concat, x1[count1:count1 + n_x1[idx]], x2[count2:count2 + n_x2[idx]]), dim=0)
        count1 += n_x1[idx]
        count2 += n_x2[idx]
    return x_concat


def unconcat_node(x, n_x1, n_x2):
    n_cum = torch.cumsum(n_x1 + n_x2, dim=0)
    n_x1a = torch.cat((torch.tensor([0]).to(x.device), n_cum))[:-1]
    n_x2a = n_x1a + n_x1
    x1 = x[n_x1a]
    x2 = x[n_x2a]
    return x1, x2


def select_graph_layer(type_model='GCN'):
    if type_model == 'GCN':
        return GCNConv
    if type_model == 'GATv2':
        return GATv2Conv
    if type_model == 'TGCN':
        return TransformerConv


class GraphLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=[32], type_model='GCN', n_heads=4,
                 skip=False, concat=False, dropout=0.5, batch_norm=True, act_func='relu'):
        super().__init__()
        assert type_model in ['GCN', 'GATv2', 'TGCN']
        self.use_batch_norm = batch_norm
        self.type_model = type_model
        self.conv = []
        self.conv_lin = []
        self.norm = []
        self.act = []
        self.p = dropout
        self.concat = concat
        self.n_heads = n_heads
        if len(hidden_channels) > 1 and skip:
            self.skip = skip
        else:
            self.skip = False
        # self.dropout = []
        for idx in range(len(hidden_channels)):
            graph_layer = select_graph_layer(self.type_model)
            if idx == 0:
                if self.type_model in ['GATv2', 'TGCN']:
                    if self.concat:
                        self.conv.append(graph_layer(in_channels, hidden_channels[idx], heads=n_heads,
                                                     concat=self.concat, edge_dim=1, dropout=self.p))
                        self.conv_lin.append(SeqLinear(self.n_heads * hidden_channels[idx],
                                                       [hidden_channels[idx]],
                                                       dropout, batch_norm, act_func))
                    else:
                        self.conv.append(graph_layer(in_channels, hidden_channels[idx], heads=n_heads,
                                                     concat=self.concat, edge_dim=1, dropout=self.p))
                else:
                    self.conv.append(graph_layer(in_channels, hidden_channels[idx],
                                                 add_self_loops=False))
            else:
                if self.type_model in ['GATv2', 'TGCN']:
                    self.conv.append(graph_layer(hidden_channels[idx - 1], hidden_channels[idx], heads=n_heads,
                                                 concat=self.concat, edge_dim=1, dropout=self.p))
                    self.conv_lin.append(SeqLinear(self.n_heads * hidden_channels[idx],
                                                   [hidden_channels[idx]],
                                                   dropout, batch_norm, act_func))
                else:
                    self.conv.append(graph_layer(hidden_channels[idx - 1], hidden_channels[idx],
                                                 add_self_loops=False))
            if idx != len(hidden_channels) - 1:
                if batch_norm:
                    # self.norm.append(nn.BatchNorm1d(hidden_channels[idx]))
                    self.norm.append(LayerNorm(hidden_channels[idx]))
                else:
                    self.norm.append(nn.Identity())
                self.act.append(get_activate_func(act_func))
                # self.dropout.append(nn.Dropout(p=dropout))

        self.conv = nn.ModuleList(self.conv)
        self.norm = nn.ModuleList(self.norm)
        # self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        self.conv_lin = nn.ModuleList(self.conv_lin)

    def forward(self, x, edge_index, edge_attr=None, batch_index=None):
        # x.shape (num_nodes in a batch, ft)
        # edge_index (2, total edges)
        # batch (num_nodes in a batch, )
        if self.type_model in ['GATv2', 'TGCN'] and edge_attr is not None:
            edge_attr = edge_attr.view(-1, 1)
        for idx in range(len(self.conv)):
            if edge_attr is not None:
                xout = self.conv[idx](x, edge_index, edge_attr)
            else:
                xout = self.conv[idx](x, edge_index)
            if self.concat and self.type_model in ['GATv2', 'TGCN']:
                xout = self.conv_lin[idx](xout)
            if self.skip:
                x = x + xout
            else:
                x = xout
            if idx != (len(self.conv) - 1):
                x = self.act[idx](x)
                x = self.norm[idx](x, batch_index)
            # x = self.dropout[idx](x)
            # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, training=self.training)

        return x, edge_index, edge_attr  # (num_nodes in a batch, hidden_channel)


class LiFu(nn.Module):
    def __init__(self, ft_trans=[768, 768], ft_gcn=[768, 512], n_heads=4, type_graph='GCN',
                 skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(LiFu, self).__init__()
        self.trans_1 = SeqLinear(768, ft_out=ft_trans,
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.trans_2 = SeqLinear(768, ft_out=ft_trans,
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.gcn = GraphLayer(in_channels=ft_trans[-1], hidden_channels=ft_gcn, type_model=type_graph, n_heads=n_heads,
                              skip=skip, concat=True, batch_norm=batch_norm, dropout=dropout, act_func=act_func)

    def forward(self, x_albef, x_dot, n_albef, n_dot, edge_index, edge_attr, batch_index):
        x_albef = self.trans_1(x_albef)  # total n_albef, ft
        x_dot = self.trans_2(x_dot)  # total n_dot, ft
        # concat x_albef + x_dot
        x_concat = concat_node(x_albef, x_dot, n_albef, n_dot)
        x, edge_index, edge_attr = self.gcn(x_concat, edge_index, edge_attr, batch_index)
        x_cls_albef, x_cls_dot = unconcat_node(x, n_albef, n_dot)
        return x_cls_albef, x_cls_dot


class EnLiFu(nn.Module):
    def __init__(self, ft_trans=[768, 768], ft_gcn=[768, 512], ft_com=[512, 512],
                 n_heads=4, type_graph='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(EnLiFu, self).__init__()
        self.gcn = LiFu(ft_trans=ft_trans, ft_gcn=ft_gcn, n_heads=n_heads, type_graph=type_graph,
                        skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
        self.lin = SeqLinear(2 * ft_gcn[-1] + 768 * 2, ft_out=ft_com, batch_norm=batch_norm,
                             dropout=dropout, act_func=act_func)

    def forward(self, x_albef, x_dot, n_albef, n_dot, edge_index, edge_attr, batch_index, x_cls_albef, x_cls_dot):
        g_cls_albef, g_cls_dot = self.gcn(x_albef, x_dot, n_albef, n_dot, edge_index, edge_attr, batch_index)
        # x_cls = torch.cat((x_cls_albef, x_cls_dot), dim=1)
        x_enc = torch.cat((g_cls_albef, x_cls_albef, x_cls_dot, g_cls_dot), dim=1)
        # x_enc = torch.cat((g_cls_albef, g_cls_dot), dim=1)
        x_enc = self.lin(x_enc)
        return x_enc


class MH(nn.Module):
    def __init__(self, ft_trans=[768], ft_gcn=[768, 512], ft_com=[512, 512],
                 n_heads=4, type_gcn='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(MH, self).__init__()
        self.enc = EnLiFu(ft_trans=ft_trans, ft_gcn=ft_gcn, ft_com=ft_com, n_heads=n_heads,
                          type_graph=type_gcn, skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)

    def forward(self, data):
        x = self.enc(x_albef=data['node_albef'], x_dot=data['node_dot'],
                     n_albef=data['n_node_albef'], n_dot=data['n_node_dot'],
                     edge_index=data['edge_index'], edge_attr=data['edge_attr'],
                     batch_index=data['batch_index'],
                     x_cls_albef=data['cls_albef'], x_cls_dot=data['cls_dot'])  # (batch, ft_com[-1])
        return x

################################################### ADAPT ##############################################
class Normalization(nn.Module):

    def __init__(self, latent_size, norm_method=None):
        super().__init__()
        if norm_method is None:
            self.norm = nn.Identity()
        elif norm_method == 'batchnorm':
            self.norm = nn.BatchNorm1d(latent_size, affine=False)
        elif norm_method == 'instancenorm':
            self.norm = nn.InstanceNorm1d(latent_size, affine=False)

    def forward(self, x):
        return self.norm(x)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(
        dim=dim, keepdim=True
    ).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s,):
    """
        Cosine similarity between all the
        image and sentence pairs
    """
    return im.mm(s.t())


class Fovea(nn.Module):

    def __init__(self, smooth=10, train_smooth=False):
        super().__init__()

        self.smooth = smooth
        self.train_smooth = train_smooth
        self.softmax = nn.Softmax(dim=-1)

        if train_smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + self.smooth)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        mask = self.softmax(x * self.smooth)
        output = mask * x
        return output


class ADAPT(nn.Module):

    def __init__(
        self, value_size, k=None, query_size=None,
        nonlinear_proj=False, groups=1,
    ):
        '''
            value_size (int): size of the features from the value matrix
            query_size (int): size of the global query vector
            k (int, optional): only used for non-linear projection
            nonlinear_proj (bool): whether to project gamma and beta non-linearly
            groups (int): number of feature groups (default=1)
        '''
        super().__init__()

        self.query_size = query_size
        self.groups = groups

        if query_size is None:
            query_size = value_size

        if nonlinear_proj:
            self.fc_gamma = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )
        else:
            self.fc_gamma = nn.Sequential(
                nn.Linear(query_size, value_size//groups),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(query_size, value_size//groups),
            )

            # self.fc_gamma = nn.Linear(cond_vector_size, in_features)
            # self.fc_beta = nn.Linear(cond_vector_size, in_features)

    def forward(self, value, query):
        '''

        Adapt embedding matrix (value) given a query vector.
        Dimension order is the same of the convolutional layers.

        Arguments:
            feat_matrix {torch.FloatTensor}
                -- shape: batch, features, timesteps
            cond_vector {torch.FloatTensor}
                -- shape: ([1 or batch], features)

        Returns:
            torch.FloatTensor
                -- shape: batch, features, timesteps

        Special cases:
            When query shape is (1, features) it is performed
            one-to-many embedding adaptation. A single vector is
            used to filter all instances from the value matrix
            leveraging the brodacast of the query vector.
            This is the default option for retrieval.

            When query shape is (batch, features) it is performed
            pairwise embedding adaptation. i.e., adaptation is performed
            line by line, and value and query must be aligned.
            This could be used for VQA or other tasks that don't require
            ranking all instances from a set.

        '''

        B, D, _ = value.shape
        Bv, Dv = query.shape

        value = value.view(
            B, D//self.groups, self.groups, -1
        )

        gammas = self.fc_gamma(query).view(
            Bv, Dv//self.groups, 1, 1
        )
        betas  = self.fc_beta(query).view(
            Bv, Dv//self.groups, 1, 1
        )

        normalized = value * (gammas + 1) + betas
        normalized = normalized.view(B, D, -1)
        return normalized


class AdaptiveEmbeddingT2I(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=1,
            gamma=10, train_gamma=False, clip_embeddings=True,
            normalization='batchnorm', use_fovea=True
        ):
        super().__init__()

        self.device = device

        self.norm = Normalization(latent_size, normalization)

        self.adapt_img = ADAPT(latent_size, k)

        self.fovea = nn.Identity()
        if use_fovea:
            self.fovea = Fovea(smooth=gamma, train_smooth=train_gamma)

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
            lens (List[int]): (B)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1).float()
        img_embed = img_embed.permute(0, 2, 1).float()

        img_embed = self.norm(img_embed)
        # cap_embed = self.norm(cap_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, cap_tensor in enumerate(cap_embed):
            # cap_tensor: 1, 1024, T
            # img_embed : B, 1024, 36
            n_words = lens[i]

            # Global textual representation
            # cap_vector: 1, 1024
            cap_repr = cap_tensor[:,:n_words].mean(-1).unsqueeze(0)

            img_output = self.adapt_img(img_embed, cap_repr)
            img_output = self.fovea(img_output)
            # Filtered global representation of the images
            img_vector = img_output.mean(-1)

            img_vector = l2norm(img_vector, dim=-1)
            cap_vector = l2norm(cap_repr, dim=-1)

            # sim = cosine_sim(img_vector, cap_vector)
            sim = cosine_sim(img_vector, cap_vector).squeeze(-1)

            # sim = sim.squeeze(-1)
            sims[:,i] = sim

        return sims


class AdaptiveEmbeddingI2T(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=1,
            gamma=1, train_gamma=False,
            normalization='batchnorm', use_fovea=True
        ):
        super().__init__()

        self.device = device

        if normalization:
            self.norm = Normalization(latent_size, normalization)

        self.adapt_txt = ADAPT(latent_size, k)

        if use_fovea:
            self.fovea = Fovea(smooth=gamma, train_smooth=train_gamma)
        else:
            self.fovea = nn.Identity()

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        #
        # cap_embed = cap_embed.permute(0, 2, 1)[...,:34].float()
        cap_embed = cap_embed.permute(0, 2, 1).float()
        img_embed = img_embed.permute(0, 2, 1).float()

        cap_embed = self.norm(cap_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        # Global image representation
        img_embed = img_embed.mean(-1)

        for i, img_tensor in enumerate(img_embed):
            # cap_tensor : B, 1024, T
            # image_embed: 1, 1024

            img_vector = img_tensor.unsqueeze(0)
            txt_output = self.adapt_txt(value=cap_embed, query=img_vector)
            txt_output = self.fovea(txt_output)

            txt_vector = txt_output.max(dim=-1)[0]

            txt_vector = l2norm(txt_vector, dim=-1)
            img_vector = l2norm(img_vector, dim=-1)
            sim = cosine_sim(img_vector, txt_vector)
            sim = sim.squeeze(-1)
            sims[i,:] = sim

        return sims


import torch.nn.functional as F


class FeatureReducer(nn.Module):
    def __init__(self, num_features, embed_dim, reduced_features):
        super(FeatureReducer, self).__init__()
        self.embed_dim = embed_dim
        self.attention = nn.Linear(embed_dim, num_features)
        self.linear = nn.Linear(num_features, reduced_features)

    def forward(self, x):
        # x: (Batch_size, num_features, embedding_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)  # (Batch_size, num_features, num_features)
        x_weighted = torch.bmm(attention_weights.transpose(1, 2), x)  # (Batch_size, num_features, embedding_dim)
        x_reduced = self.linear(x_weighted.transpose(1, 2))  # (Batch_size, embedding_dim, reduced_features)
        return x_reduced.transpose(1, 2)  # (Batch_size, reduced_features, embedding_dim)


class AttentionFlattener(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(AttentionFlattener, self).__init__()
        self.attention = nn.Linear(embed_dim, num_features)
        self.transform_2 = nn.Linear(embed_dim, 1024)

    def forward(self, x):
        # x: (Batch_size, num_features, embedding_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)  # (Batch_size, num_features, num_features)
        x_weighted = torch.bmm(attention_weights.transpose(1, 2), x)  # (Batch_size, num_features, embedding_dim)
        x_flattened = x_weighted.mean(dim=1)  # (Batch_size, embedding_dim)
        x_flattened = self.transform_2(x_flattened)

        return x_flattened
