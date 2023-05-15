
from turtle import position
import torch
import torch.nn as nn
from t_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        pad_index, # TODO: use to make a padding mask....
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.pad_index = pad_index
        self.device = device
        self.word_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dim_out = nn.Linear(embed_size, input_vocab_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, input):

        N, input_len = input.shape

        input_mask = (input != self.pad_index).unsqueeze(1).unsqueeze(2)

        mask = torch.tril(torch.ones((input_len, input_len))).to(self.device)
        mask = mask.masked_fill(input_mask == 0, 0)

        # this is the appropriate dimension for multi-head expansion
        mask = mask.expand(
            N, 1, input_len, input_len
        )
        return mask.to(self.device)

    def embed_input(self, input):

        num_ex, seq_len = input.shape
        position_emb = torch.arange(0, seq_len).expand(num_ex, seq_len).to(self.device)
        embeddings = self.word_embedding(input)
        p_embeddings = self.position_embedding(position_emb)

        return (embeddings + p_embeddings)

    def forward(self, input):
        mask = self.make_mask(input)
        x = self.embed_input(input)
        x = self.dropout(x)

        # this looks funny because the block was built to use with encoder/decoder model
        for layer in self.layers:
            x = layer(x, x, x, mask)

        out = self.dim_out(x)

        return out
