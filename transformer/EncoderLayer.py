from torch import nn

from transformer.MutliHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, feed_forward_size, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        out = self.layer_norm(x + attn_output)
        ffn_output = self.ffn(out)
        ffn_output = self.dropout(ffn_output)
        out = self.layer_norm(out + ffn_output)
        return out