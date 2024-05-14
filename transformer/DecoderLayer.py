from torch import nn

from transformer.MutliHeadAttention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, feed_forward_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.src_attn = MultiHeadAttention(embed_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x, src, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        src_attn_output = self.src_attn(x, src, src, src_mask)
        ffn_output = self.ffn(self.layer_norm(self_attn_output + src_attn_output))
        return self.dropout(ffn_output)