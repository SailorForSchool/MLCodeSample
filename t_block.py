'''
Notes on the project as a whole:

    - each of the functions defined has a seperate values, keys, query value and
    this is in order to use the encoder decoder model where half of the model
    'encodes' and is used as the input for attention in the 'decode' part of the
    model. In a transformer that is not of this type, it will simply be the same
    input, the word embedding, that is used to find the different values for v, k,
    and q in the linear layers

'''

import torch
import torch.nn as nn

class mult_h_attention(nn.Module):

    def __init__(self, embed_size, num_heads):
        # initialize parent class
        super(mult_h_attention, self).__init__()

        # record variables for use during attention
        self.embed_size = embed_size
        self.num_heads = num_heads

        # this determines the size of each head since they are concat later
        self.head_dim = embed_size // num_heads
        ''' NOTE: head_dim should be a perfect division of the embed_dim'''
        assert (
                self.head_dim * num_heads == embed_size

        ), "Embedding size must be compatible with number of heads expected"

        # no bias here for now.... TODO
        # this is the learnable parameters for Value, Query, and Key
        self.tovalues = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.toqueries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.tokeys = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # define the multi head output to be of the original data dimensions
        self.unify_heads = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):

        # number of training examples given to the model
        num_ex = query.shape[0]

        # dimensions of Q, V, K
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        # now we run through the linear layers...
        # and split into heads...

        # get v vectors
        values = values.reshape(num_ex, value_len, self.num_heads, self.head_dim)
        values = self.tovalues(values)
        # get k_i vectors
        keys = keys.reshape(num_ex, key_len, self.num_heads, self.head_dim)
        keys = self.tokeys(keys)
        # get q vectors
        query = query.reshape(num_ex, query_len, self.num_heads, self.head_dim)
        query = self.toqueries(query)
        
        # Einsum matrix multiplication to resolve query key similarity
        dot = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        '''
        queries shape: (num_ex, query_len, num_heads, head_dim),
        keys shape: (num_ex, key_len, num_heads, head_dim)
        dot: (num_ex, num_heads, query_len, key_len)
        '''

        # mask padded indices so that the weights are 0 after Softmax
        # NOTE: this is either the sentence padding for <pad> or causal
        # attention

        if mask is not None:
            dot = dot.masked_fill(mask == 0, float(-1e20))

        # compute attention matrix
        # note: dim = 3 because of input dimensions
        attention = torch.softmax(dot / (self.embed_size ** (1/2)), dim=3)

        # Einsum matrix multiplication to resolve new representations using v vectors
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        '''
        attention shape: (num_ex, num_heads, query_len, key_len)
        values shape: (num_ex, value_len, num_heads, head_dim)
        out shape: (num_ex, query_len, num_heads, head_dim)
        '''

        # reshape / flatten heads, self.num_heads * self.head_dim = embed_dim
        out = out.reshape(num_ex, query_len, self.num_heads * self.head_dim)
        
        return self.unify_heads(out)

class TransformerBlock(nn.Module):

    def __init__(self, embed_size, num_heads, dropout, forward_exp):
        super(TransformerBlock, self).__init__()
        self.attention = mult_h_attention(embed_size, num_heads)
        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)

        # single layer MLP
        self.feed_forward = nn.Sequential(
            # forward_exp determines the width of the MLP layer
            nn.Linear(embed_size, forward_exp * embed_size),
            nn.ReLU(), # here is our non-linearity!
            nn.Linear(forward_exp * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # attention...
        attention = self.attention(value, key, query, mask)
   
        # dropout and layernorm prevent over fitting (theoretically)
        x = self.dropout(self.norm_1(attention + query))

        # ... MLP ...
        forward = self.feed_forward(x)
        out = self.dropout(self.norm_2(forward + x))

        return out
