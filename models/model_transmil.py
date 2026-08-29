import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TransMIL: Transformer based Correlated Multiple Instance Learning
Shao et al. -- NeurIPS 2021, arXiv:2106.00908

Faithful to the reference implementation (github.com/szc19990412/TransMIL):
    features -> fc-512 + ReLU -> squaring of the sequence -> class token
             -> TransLayer (MSA, correlation modelling)
             -> PPEG (pyramid position encoding)
             -> TransLayer (MSA, deep aggregation)
             -> LayerNorm -> class token -> Linear(512, n_classes)

The 512 above is the paper's width, kept by --model_size small/big; tiny and supertiny
narrow the whole trunk (see size_dict).
"""


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        try:
            from nystrom_attention import NystromAttention
        except ImportError:
            raise ImportError(
                "TransMIL needs the nystrom-attention package (pip install nystrom-attention). "
                "The Nystrom approximation is what keeps attention linear in the number of "
                "patches; plain O(n^2) attention is not viable on bags of thousands of patches.")

        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout = 0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))   # pre-norm residual
        return x


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator: depthwise convs with kernels 7, 5, 3 on the
    tokens reshaped back to a 2D grid, summed with the identity."""
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes = 2, embed_dim = 1024, dropout = 0., size_arg = "small"):
        super(TransMIL, self).__init__()
        # adaptation 4: the reference implementation hardcodes a 512-wide trunk in six
        # places. small/big keep it; tiny and supertiny narrow it for the capacity
        # ablation. NystromAttention derives dim_head = dim//8 across 8 heads, so dim
        # must stay a multiple of 8 -- 512, 256 and 128 all do.
        self.size_dict = {"small": 512, "big": 512, "tiny": 256, "supertiny": 128}
        dim = self.size_dict[size_arg]

        self.pos_layer = PPEG(dim=dim)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, dim), nn.ReLU())   # adaptation 3
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim)
        self.layer2 = TransLayer(dim=dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if h.dim() == 2:            # adaptation 1: [N, D] -> [1, N, D]
            h = h.unsqueeze(0)

        h = self._fc1(h)            # [B, n, 512]

        # squaring of the sequence: pad to the next perfect square by repeating the
        # first instances, so the tokens can be laid out on a 2D grid for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)   # adaptation 2
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)                 # MSA: correlation modelling
        h = self.pos_layer(h, _H, _W)      # PPEG
        h = self.layer2(h)                 # MSA: deep aggregation

        h = self.norm(h)[:, 0]             # class token
        logits = self._fc2(h)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': h})

        # no per-instance attention map: attention lives inside the Nystrom blocks
        return logits, Y_prob, Y_hat, None, results_dict
