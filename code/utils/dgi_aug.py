import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def drop_edge(adj_matrix, drop_prob=0.2):
    if isinstance(adj_matrix, sp.coo_matrix):
        adj_matrix = torch.FloatTensor(adj_matrix.toarray())
    mask = torch.rand_like(adj_matrix).to(torch.float32)
    mask = mask.bernoulli(p=1 - drop_prob)
    adj_matrix = adj_matrix * mask
    return adj_matrix

def feature_masking(features, mask_prob=0.2):
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    mask = torch.rand_like(features).to(torch.float32)
    mask = mask.bernoulli(p=1 - mask_prob)
    features = features * mask
    return features.numpy()

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.FloatTensor(out_ft)) if bias else None
        if bias:
            self.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.counter = 0

def dgi_embed(adj, features, hid_units, model_name, cuda=False, opt=None):
    adj = sp.coo_matrix(adj)
    adj = normalize_adj_dgi(adj + sp.eye(adj.shape[0]))
    if opt and opt.use_drop_edge == 1:
        print("Applying DropEdge augmentation...")
        adj = drop_edge(adj, drop_prob=opt.drop_prob)
    if opt and opt.use_feature_masking == 1:
        print("Applying Feature Masking augmentation...")
        features = feature_masking(features, mask_prob=opt.mask_prob)
    features = torch.FloatTensor(features[np.newaxis])
    adj = sparse_mx_to_torch_sparse_tensor(adj) if opt.sparse else torch.FloatTensor(adj[np.newaxis])
    model = DGI(features.shape[2], hid_units, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
    for epoch in range(400):
        model.train()
        optimiser.zero_grad()
        idx = np.random.permutation(features.shape[1])
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(1, features.shape[1])
        lbl_2 = torch.zeros(1, features.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1)
        logits = model(features, shuf_fts, adj, opt.sparse, None, None, None)
        loss = nn.BCEWithLogitsLoss()(logits, lbl)
        loss.backward()
        optimiser.step()
        if epoch % 10 == 0:
            print('Epoch:', epoch, 'Loss:', loss.item())
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    embeds, _ = model.embed(features, adj, opt.sparse, None)
    return embeds[0].numpy()

def normalize_adj_dgi(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
