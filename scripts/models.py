import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # queries, keys, values weight matrices
        self.atten_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.atten_proj(x).split(self.embed_dim, dim=-1)
        b, s, _ = Q.shape

        Q = Q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        atten = (
            F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout, is_causal=True
            )
            .transpose(1, 2)
            .contiguous()
            .view(b, s, -1)
        )
        output = F.dropout(self.output_proj(atten), p=self.dropout).contiguous()
        return output


class MLPLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = F.dropout(self.fc2(x), self.dropout)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn_layer = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc_layer = MLPLayer(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn_layer(self.ln1(x))
        x = x + self.fc_layer(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        block_size,
        vocab_size,
        num_heads,
        num_blocks,
        dropout=0.0,
        PADDING_TOKEN_ID=None,
        BOS_TOKEN_ID=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.PADDING_TOKEN_ID = PADDING_TOKEN_ID
        self.BOS_TOKEN_ID = BOS_TOKEN_ID

        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeds = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):

        b, s = x.shape
        seq_len = torch.arange(s, dtype=torch.long, device=x.device).unsqueeze(0)

        token_embed = self.token_embeds(x)
        pos_embed = self.pos_embeds(seq_len)
        x = self.dropout(token_embed + pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        if targets is not None:
            preds = self.lm_head(x)
            # exclude <BOS> token from predictions
            preds[:, :, self.BOS_TOKEN_ID] += float("-inf")

            # compute loss
            loss = F.cross_entropy(
                preds.reshape(-1, preds.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
                ignore_index=self.PADDING_TOKEN_ID,
            )

        else:

            preds = self.lm_head(x[:, [-1], :])
            preds[:, :, self.BOS_TOKEN_ID] += float("-inf")
            loss = None

        return preds, loss
