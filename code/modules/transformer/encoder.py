import torch
import torch.nn as nn
import math
import numpy as np
from code.modules.transformer.layer import EncoderLayer

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoder_(nn.Module):
    r"""
        d_word_vec: int -> size of embedded sequence
        n_layers: int -> number of layer
        n_head: int
        d_model: int 
        pad_idx: int

    """
    def __init__(
            self, d_word_vec, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, scale_emb=False, dropout=0.1):

        super().__init__()

        # self.layer_stack = nn.ModuleList([
        #     EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        #     for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.scale_emb = scale_emb
        # self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, embedded_seq, src_mask, return_attns=False):

        r"""
            embed_seq: tensor[batch, n_seq, d_word_vec]
        """

        # print(embedded_seq.shape)
        # print(src_mask.shape)

        # enc_slf_attn_list = []

        # # -- Forward
        # enc_output = embedded_seq

        # for enc_layer in self.layer_stack:
        #     enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        #     enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # if return_attns:
        #     return enc_output, enc_slf_attn_list
        # return enc_output,

        src = embedded_seq * math.sqrt(self.d_model)
        if not self.pos_encoder is None:
            src = self.pos_encoder(src)
        
        return self.transformer_encoder(src, src_key_padding_mask=src_mask)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)