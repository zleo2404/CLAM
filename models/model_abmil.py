import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_clam import Attn_Net, Attn_Net_Gated

"""
Attention-based Deep Multiple Instance Learning (ABMIL)
Ilse, Tomczak, Welling -- ICML 2018, arXiv:1802.04712

Attention pooling (eq. 8):
    a_k = softmax_k( w^T tanh(V h_k^T) )                      V in R^{LxM}, w in R^{Lx1}
Gated attention pooling (eq. 9):
    a_k = softmax_k( w^T (tanh(V h_k^T) * sigm(U h_k^T)) )    U in R^{LxM}
    z   = sum_k a_k h_k

ABMIL reuse CLAM istructions since its tha same but without the instance-level clustering branch.
"""


class ABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., n_classes = 2, embed_dim = 1024):
        super().__init__()
        # [input dim, embedding dim M, attention hidden dim L]
        # L=128 is the paper's "mil-attention-128"
        self.size_dict = {"small": [embed_dim, 512, 128], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
              nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)       # A: N x 1, h: N x M
        A = torch.transpose(A, 1, 0)       # 1 x N
        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)            # softmax over the instances of the bag

        M = torch.mm(A, h)                 # z = sum_k a_k h_k
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict
