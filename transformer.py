import math
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        # Calculate the dot product of q and k, and scale it by the temperature
        # Then apply the dropout layer and the softmax function to the result
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        
        # Calculate the dot product of the attention weights and v
        output = torch.matmul(attn, v)

        # Return the attention-weighted output
        return output


class MultiHeadCrossAttention(nn.Module):
    ''' Multi-Head Cross Attention module '''
    
    def __init__(self, n_token, n_head, d_model, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.n_token = n_token
        self.n_head = n_head
        
        self.d_k = d_k
        self.d_v = d_v
        
        # Create the query token and apply uniform initialization
        self.q = nn.Parameter(torch.empty((1, n_token, d_model)))
        q_init_val = math.sqrt(1 / d_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)
        
        # Create linear layers for the query, key, and value projections
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        # Create a linear layer for the final projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        # Create a ScaledDotProductAttention layer
        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5,
            attn_dropout=attn_dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        n_token = self.n_token
        b, len_seq = x.shape[:2]
        
        # Project and reshape
        q = self.w_qs(self.q).view(1, n_token, n_head, d_k)
        k = self.w_ks(x).view(b, len_seq, n_head, d_k)
        v = self.w_vs(x).view(b, len_seq, n_head, d_v)

        # Transpose to (batch_size, n_head, sequence_length, d_k/d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply the attention mechanism
        x = self.attention(q, k, v)
        
        # Transpose back to (batch_size, sequence_length, n_head*d_v)
        x = x.transpose(1, 2).contiguous().view(b, n_token, -1)
        
        # Apply the final projection, dropout, and residual connection
        x = self.dropout(self.fc(x))
        x += self.q
        
        # Apply layer normalization
        x = self.layer_norm(x)

        return x


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        
        # Create linear layers for the feed-forward network
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        
        # Apply both layers
        x = self.w_2(torch.relu(self.w_1(x)))
        
        # Apply dropout and add the residual
        x = self.dropout(x)
        x += residual
        
        # Apply layer normalization
        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    """ Composition of Cross-Attention and MLP """
    
    def __init__(self, n_token, d_model, d_inner, n_head, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.n_token = n_token
        
        # Create a MultiHeadCrossAttention layer
        self.crs_attn = MultiHeadCrossAttention(n_token, n_head, d_model, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
        
        # Create a PositionwiseFeedForward layer
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, x):
        # Apply the cross-attention mechanism
        x = self.crs_attn(x)
        
        # Apply the position-wise feed-forward network
        x = self.pos_ffn(x)

        return x


class Transformer(nn.Module):
    """ Transformer architecture """
    
    def __init__(self, n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        # Create a list of EncoderLayer modules, n_layer=1 should suffice
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_token[i], d_model, d_inner, n_head, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
            for i in range(n_layer)])
    
    def forward(self, x):
        
        # Iterate over the encoder layers in the layer stack
        for enc_layer in self.layer_stack:
            # Apply the encoder layer to the input tensor
            x = enc_layer(x)

        return  x
