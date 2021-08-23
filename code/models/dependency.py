# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from code.modules import (LSTM, MLP, BertEmbedding, Biaffine, CharLSTM,
                           Triaffine)
from code.modules.dropout import IndependentDropout, SharedDropout
from code.utils import Config
from code.utils.alg import eisner, eisner2o, mst
from code.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from code.modules import TransformerEncoder_


class BiaffineDependencyModel(nn.Module):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_feats (int):
            The size of the feat vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        feat (str):
            Specifies which type of additional feature to use: ``'char'`` | ``'bert'`` | ``'tag'``.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            ``'tag'``: POS tag embeddings.
            Default: ``'char'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_arc (int):
            Arc MLP size. Default: 500.
        n_mlp_rel  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_rels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 encoder='bert',
                 transformer_n_layers=6,
                 transformer_n_head=8,
                 transformer_d_k=64,   
                 transformer_d_v=64,
                 transformer_d_inner=2048,
                 transformer_scale_embed=False,
                 transformer_dropout=0.1,
                 n_feats_pos = None,   
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        print('encoder', encoder)

        self.encoder_type = encoder
        self.feat_type = feat

        # the embedding layer

        print('n_feats_pos', n_feats_pos)

        if encoder == 'lstm':

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

            self.lstm = LSTM(input_size=n_embed + n_feat_embed,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
            self.lstm_dropout = SharedDropout(p=lstm_dropout)

            self.encoder_n_out = n_lstm_hidden*2

        elif encoder == 'bert':

            self.encoder = BertEmbedding(model=bert,
                                         n_layers=n_bert_layers,
                                         pad_index=pad_index,
                                         dropout=mix_dropout,
                                         requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=lstm_dropout)

            if n_feats_pos is not None:
                self.encoder_n_out = (self.encoder.n_out + n_feats_pos)
            else:
                self.encoder_n_out = self.encoder.n_out

        elif encoder == 'transformer':

            #cần thêm positional embedding để test

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

            self.transformer_encoder = TransformerEncoder_(d_word_vec=n_embed + n_feat_embed,
                                                        n_layers=transformer_n_layers,
                                                        n_head=transformer_n_head,
                                                        d_k=transformer_d_k,   
                                                        d_v=transformer_d_v,
                                                        d_model=n_embed + n_feat_embed,
                                                        d_inner=transformer_d_inner,
                                                        scale_emb=False,
                                                        dropout=0.1)


            self.encoder_dropout = nn.Dropout(p=lstm_dropout)
            
            self.encoder_n_out = n_embed + n_feat_embed
        

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=self.encoder_n_out, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=self.encoder_n_out, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.encoder_n_out, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.encoder_n_out, n_out=n_mlp_rel, dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def freeze(self):
        for name, child in self.named_children():
            if self.feat_type == 'bert' and name == 'feat_embed':
                continue
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for name, child in self.named_children():
            if self.feat_type == 'bert' and name == 'feat_embed':
                continue
            for param in child.parameters():
                param.requires_grad = True

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            #nn.init.zeros_(self.word_embed.weight)
            nn.init.orthogonal_(self.word_embed.weight) # use orthogonal matrix initialization
        return self

    def encode(self, words, feats=None, embedded_pos=None):

        #print('words shape', words.shape)

        if self.encoder_type == 'bert':
            x = self.encoder(words)
            x = self.encoder_dropout(x)
            embed = torch.cat((x, embedded_pos), -1)
            return embed
        else:
            #embedding phase

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
            feat_embed = self.feat_embed(feats)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), -1)

            if self.encoder_type == 'lstm':
                x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
                x, _ = self.lstm(x)
                x, _ = pad_packed_sequence(x, True, total_length=seq_len)
                x = self.lstm_dropout(x)

                return x
            else:
                key_padding_mask = ~mask

                #print('mask shape', src_mask.shape)

                x = self.transformer_encoder(embed, key_padding_mask)
                x = self.encoder_dropout(x)

                # with open('test.txt', 'a') as f:
                #     print(x, file=f)

                return x

    def forward(self, words, feats, embedded_pos=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (~torch.LongTensor):
                Feat indices.
                If feat is ``'char'`` or ``'bert'``, the size of feats should be ``[batch_size, seq_len, fix_len]``.
                if ``'tag'``, the size is ``[batch_size, seq_len]``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """

        # if self.encoder_type != 'bert':
        #     batch_size, seq_len = words.shape

        
        # get the mask and lengths of given batch
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        x = self.encode(words, feats, embedded_pos)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class AffineDependencyModel(nn.Module):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_feats (int):
            The size of the feat vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        feat (str):
            Specifies which type of additional feature to use: ``'char'`` | ``'bert'`` | ``'tag'``.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            ``'tag'``: POS tag embeddings.
            Default: ``'char'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_arc (int):
            Arc MLP size. Default: 500.
        n_mlp_rel  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_feats_add,
                 n_rels_add,
                 n_feat_embed=100,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 pad_index_add=0,
                 unk_index_add=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        
        self.feat_embed = nn.Embedding(num_embeddings=n_feats_add,
                                           embedding_dim=n_feat_embed)
       
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = LSTM(input_size=n_feat_embed,
                         hidden_size=n_lstm_hidden,
                         num_layers=n_lstm_layers,
                         bidirectional=True,
                         dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden*2, n_out=n_mlp_rel, dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels_add, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index_add
        self.unk_index = unk_index_add

    def freeze(self):
        # for param in self.parameters():
        #     param.requires_grad = False
        for n, child in self.named_children():
            if n == 'feat_embed' or n == 'embed_dropout':
                continue
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            #nn.init.zeros_(self.feat_embed.weight)
            nn.init.orthogonal_(self.word_embed.weight) # use orthogonal matrix initialization
        return self

    def encode(self, words):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.feat_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        feat_embed = self.feat_embed(ext_words)
        if hasattr(self, 'pretrained'):
            feat_embed += self.pretrained(words)
        feat_embed = self.embed_dropout(feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((feat_embed[0], ), -1)
        return embed

    def forward(self, words):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (~torch.LongTensor):
                Feat indices.
                If feat is ``'char'`` or ``'bert'``, the size of feats should be ``[batch_size, seq_len, fix_len]``.
                if ``'tag'``, the size is ``[batch_size, seq_len]``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.feat_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        feat_embed = self.feat_embed(ext_words)
        if hasattr(self, 'pretrained'):
            feat_embed += self.pretrained(words)
        feat_embed = self.embed_dropout(feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((feat_embed[0], ), -1)
        
        x = pack_padded_sequence(embed, mask.sum(1).tolist(), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        # s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds