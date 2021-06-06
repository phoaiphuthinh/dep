import torch 
import torch.nn as nn
from code.modules import (LSTM, MLP, BertEmbedding, Biaffine, CharLSTM,
                           Triaffine)
from code.modules.dropout import IndependentDropout, SharedDropout
from code.modules import Convert
from code.utils import Config
from code.utils.alg import eisner, eisner2o, mst
from code.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Guessing_Layer(nn.Module):
    def __init__(self, n_in, n_out, pad_index, dropout=0):
        super().__init__()

        self.n_hidden = 256

        self.layer_linear = MLP(n_in, self.n_hidden)
        self.layer_linear_2 = MLP(self.n_hidden, n_out)
        self.n_out = n_out
        self.pad_index = pad_index

    def forward(self, x, mask):
        
        res = self.layer_linear(x)
        res = self.layer_linear_2(res)
        #res = self.layer_relu(res)
        # mask_embed = torch.tile(mask, (1, self.n_out))

        # res = res.masked_filled_(mask_embed, self.pad_index)

        return res
    
    def _freeze(self):
        self.layer_linear.eval()
        self.layer_linear_2.eval()
        for param in self.layer_linear.parameters():
            param.requires_grad = False
        for param in self.layer_linear_2.parameters():
            param.requires_grad = False

    def _unfreeze(self):
        self.layer_linear.train()
        self.layer_linear_2.train()
        for param in self.layer_linear.parameters():
            param.requires_grad = True
        for param in self.layer_linear_2.parameters():
            param.requires_grad = True