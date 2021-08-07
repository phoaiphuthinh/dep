import torch
import torch.nn as nn
import numpy as np
from code.modules.transformer.layer import EncoderLayer

class Encoder(nn.Module):
    r"""
        d_word_vec: int -> size of embedded sequence
        n_layers: int -> number of layer
        n_head: int
        d_model: int 
        pad_idx: int

    """
    def __init__(
            self, d_word_vec, n_layers, n_head, d_k=64, d_v=64,
            d_model=512, d_inner=2048, pad_idx=0, scale_emb=False, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, embedded_seq, src_mask, return_attns=False):

        r"""
            embed_seq: tensor[batch, n_seq, d_word_vec]
        """

        enc_slf_attn_list = []

        # -- Forward
        enc_output = embedded_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,