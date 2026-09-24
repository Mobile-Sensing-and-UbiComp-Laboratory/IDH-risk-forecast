import os
import torch
import torch.nn as nn
import math
from einops import rearrange
import numpy as np

DEVICE = torch.device(os.environ.get('IDH_DEVICE', 'cuda'))
print("DEVICE:", DEVICE)

# =========== HELPER LAYER ========================================================================
class CheckShape(nn.Module):
    def __init__(self, remark, key=None):
        super().__init__()
        self.remark = remark
        self.key = key
    def forward(self, x):
        if self.remark is not None:
            print(self.remark, x.shape)
        
        out = x
        if self.key is not None:
            out = self.key(x)
        return out
    
class VAE_Latent(nn.Module):
    def __init__(self, emb_size, out_size):
        super().__init__()

        self.mu = nn.Linear(emb_size, out_size)
        self.var = nn.Sequential(
            nn.Linear(emb_size, out_size),
            nn.Softplus()
        )
        
    def forward(self, x, latent_only=False):
        # generate mean and variance
        mu, var = self.mu(x), self.var(x)

        # reparametrization trick
        if self.training:
            eps = torch.randn_like(var).to(DEVICE)
            z = mu + var*eps
        else:
            z = mu
        
        # output
        if latent_only:
            return z
        return z, mu, var

# =========== Encoder Block ========================================================================
class RNNEncoder(nn.Module):
    def __init__(
        self,
        in_channel, # 22
        emb_size=64,
    ):
        super().__init__()

        self.rnn = nn.LSTM(in_channel, emb_size, 1, batch_first=True)
        

    def forward(self, x):
        out, (_, _) = self.rnn(x.squeeze(1))
        return out
    
class EmbConvBlock(nn.Module):
    def __init__(
        self, 
        in_channel, # 22
        T_kernel_size, # 8,
        emb_size=64,
        hidden_size=64*4
    ):
        super().__init__()
        
        # Input shape: (N, L, C)
        self.liner = nn.Sequential(
            CheckShape(None, key=lambda x: x.unsqueeze(1)), #(N, 1, L, C)
            # Temporal
            nn.Conv2d(1, hidden_size, kernel_size=[T_kernel_size, 1], padding='same'), 
            # Encoder variant: dilated temporal convolution instead of the active convolution.
            # nn.Conv2d(1, emb_size, kernel_size=[T_kernel_size, 1], padding='same', dilation=2), # no warning
            nn.BatchNorm2d(hidden_size), 
            nn.GELU(),
            # Spatial
            nn.Conv2d(hidden_size, emb_size, kernel_size=[1, in_channel], padding='valid'), 
            nn.BatchNorm2d(emb_size), 
            nn.GELU(),
            CheckShape(None, key=lambda x: torch.permute(x, (0, 3, 2, 1)).squeeze(1)), # (N, L, C)
        )

    def forward(self, x):
        # Input shape: (N, L, C)
        out = self.liner(x)
        return out
    
# =========== MAIN TRANSFORMER LAYERS ========================================================================
class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len=22, dropout=0.5, add_norm=True):
        super().__init__()
        self.add_norm = add_norm
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(p=dropout)
        self.to_out = nn.LayerNorm(emb_size)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # compute key, query, value vectors
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # compute attention
        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        # add bias
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # final out
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        # Add & Norm
        if self.add_norm:
            return self.LayerNorm(x + out)
        else:
            return out

######## ViT version attention ########
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(nn.functional.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return self.norm(self.drop(self.proj(h))+x)
################################################

class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True):
        super().__init__()
        self.add_norm = add_norm

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            # Regularization variant: add dropout at this point in the feed-forward block.
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out
    
class Transformer(nn.Module):
    def __init__(
        self, 
        num_layers=1,
        emb_size=16,
        num_heads=8,
        hidden_size=int(16*4),
        dropout=0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                MultiHeadedSelfAttention(emb_size, num_heads, dropout),
                FeedForward(emb_size, hidden_size, dropout=dropout),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Input shape: (N, L, C)
        for block in self.blocks:
            x = block(x)
        return x # (N, L, E)
    
class SiT(nn.Module):
    def __init__(
        self, 
        num_layers=12,
        emb_size=768,
        num_heads=12,
        dropout=0.1,
        # data related
        in_channel=8,
        seq_length=256,
    ):
        super().__init__()

        self.emb = nn.Sequential( # (N, L, C)
            # Patching
            EmbConvBlock(in_channel, 8, emb_size, hidden_size=emb_size*4),
            # position embedding
            tAPE(emb_size, dropout=dropout, max_len=seq_length),
        ) # (N, L, E)

        # transformers blocks
        self.transformer = Transformer(
            num_layers=num_layers,
            emb_size=emb_size,
            num_heads=num_heads,
            hidden_size=emb_size*4,
            dropout=dropout,
        )

    def forward(self, x): # Input shape: (N, L, C)
        out = self.emb(x)
        out = self.transformer(out) 
        return out # (N, L, E)


