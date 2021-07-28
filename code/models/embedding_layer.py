import torch
import torch.nn as nn
from code.modules import (LSTM, MLP, BertEmbedding, Biaffine, CharLSTM,
                           Triaffine)
from code.modules.dropout import IndependentDropout, SharedDropout
from code.utils import Config
from code.utils.alg import eisner, eisner2o, mst
from code.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Embedding_Layer(nn.Module):
    def __init__(self, 
                n_words,
                n_feats,
                feat='char',
                alpha=0.3,
                n_embed=100,
                n_feat_embed=100,
                n_feats_add=20,
                n_char_embed=50,
                bert=None,
                n_bert_layers=4,
                mix_dropout=.0,
                embed_dropout=.33,
                feat_pad_index=0,
                pad_index=0,
                pad_index_add=0,
                unk_index=1,
                unk_index_add=1,
                **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        #print('feat: ', feat)

        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")

        self.embed_dropout = IndependentDropout(p=embed_dropout)

        self.pad_index = pad_index
        self.unk_index = unk_index


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
            #nn.init.orthogonal_(self.word_embed.weight) # use orthogonal matrix initialization
        return self

    def _freeze(self):
        self.word_embed.eval()
        for params in self.word_embed.parameters():
            params.requires_grad = False
    
    def _unfreeze(self):
        self.word_embed.train()
        for params in self.word_embed.parameters():
            params.requires_grad = True

    def forward(self, words, feats=None):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers

        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats) if feats is not None else None
        if feats is not None:
            #word_embed += feat_embed
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed) 
            word_embed += feat_embed
        else:
            word_embed = self.embed_dropout(word_embed)
            word_embed = word_embed[0]

        # with open("test.txt", "a+") as f:
        #     print(mask, file=f)
        return word_embed, feat_embed, mask, seq_len

        