from torch import nn

from transformer.EncoderLayer import EncoderLayer
from transformer.PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, num_layers, embed_size, num_heads, feed_forward_size, dropout, max_len=5000):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_size, num_heads, feed_forward_size, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

    def forward(self, x, mask):
        N, T, _ = x.shape
        pos_encoding = self.positional_encoding(x)
        x = pos_encoding
        for layer in self.layers:
            x = layer(x, mask)
        return self.dropout(x)