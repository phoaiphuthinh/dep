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
from code.models.dependency import BiaffineDependencyModel, AffineDependencyModel


class EnsembleModel(nn.Module):


    def __init__(self, 
                n_words,
                n_feats,
                n_rels,
                n_rels_add,
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
                n_lstm_hidden=400,
                n_lstm_layers=3,
                lstm_dropout=.33,
                n_mlp_arc=500,
                n_mlp_rel=100,
                mlp_dropout=.33,
                feat_pad_index=0,
                pad_index=0,
                pad_index_add=0,
                unk_index=1,
                unk_index_add=1,
                **kwargs):

        super().__init__()

        self.args = Config().update(locals())

        print(self.args)

        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)

        

        self.origin = BiaffineDependencyModel(  n_words=n_words,
                                                n_feats=n_feats,
                                                n_rels=n_rels,
                                                feat=feat,
                                                n_embed=n_embed,
                                                n_feat_embed=n_feat_embed,
                                                n_char_embed=n_char_embed,
                                                bert=bert,
                                                n_bert_layers=n_bert_layers,
                                                mix_dropout=mix_dropout,
                                                embed_dropout=embed_dropout,
                                                n_lstm_hidden=n_lstm_hidden,
                                                n_lstm_layers=n_lstm_layers,
                                                lstm_dropout=lstm_dropout,
                                                n_mlp_arc=n_mlp_arc,
                                                n_mlp_rel=n_mlp_rel,
                                                mlp_dropout=mlp_dropout,
                                                feat_pad_index=feat_pad_index,
                                                pad_index=pad_index,
                                                unk_index=unk_index) #More argument
        self.addition = AffineDependencyModel(n_feats_add=n_feats_add,
                                                n_rels_add=n_rels_add,
                                                embed_dropout=embed_dropout,
                                                n_lstm_hidden=n_lstm_hidden,
                                                n_lstm_layers=n_lstm_layers,
                                                lstm_dropout=lstm_dropout,
                                                n_mlp_arc=n_mlp_arc,
                                                n_mlp_rel=n_mlp_rel,
                                                mlp_dropout=mlp_dropout,
                                                pad_index_add=pad_index_add,
                                                unk_index_add=unk_index_add) #More

        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.alpha = alpha
        self.n_rels_add = n_rels
        self.softmax_arc = nn.Softmax(dim=1)
        self.softmax_rel = nn.Softmax(dim=3)
        #self.redirect


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            # nn.init.zeros_(self.word_embed.weight)
            nn.init.orthogonal_(self.word_embed.weight) # use orthogonal matrix initialization
        return self
    

    def forward(self, words, feats, adds=None, pos=None):

        s_arc, s_rel = self.origin(words, feats)
        if adds is not None:
            a_arc, a_rel = self.addition(adds)
            #a_arc: [bucket_size, seq_len, seq_len]
        
            self.modifyScore_3(adds, a_arc, a_rel, pos, s_arc, s_rel)
            #print(s_arc.shape)

            # s_arc = self.softmax_1(s_arc)
            # s_rel = self.softmax_1(s_rel)

            # a_arc = self.softmax_1(a_arc)
            # a_rel = self.softmax_1(a_rel)
            
            return s_arc, s_rel, a_arc, a_rel

        a_arc, a_rel = self.addition(pos)
    
        self.modifyScore_3(pos, a_arc, a_rel, pos, s_arc, s_rel)

            
        return s_arc, s_rel


    def loss(self, s_arc, s_rel, arcs, rels, mask, a_arc=None, a_rel=None, arcs_add=None,rels_add=None, mask_add=None, partial=False):
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
        if mask_add is not None:
            return self.origin.loss(s_arc, s_rel, arcs, rels, mask) * self.addition.loss(a_arc, a_rel, arcs_add, rels_add, mask_add)

        return self.origin.loss(s_arc, s_rel, arcs, rels, mask)

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

    # def modifyscore(self, adds, a_arc, a_rel, pos, s_arc, s_rel):

    #     score_arc = {}
    #     score_rel = {}

    #     bucket_size, seq_len = adds.shape

    #     for i, buck in enumerate(a_arc):
    #         sen = adds[i]
    #         for p1 in range(seq_len):
    #             for p2 in range(seq_len):
    #                 tup = (sen[p1], sen[p2])
    #                 score_arc[tup] = score_arc.get(tup, 0) + buck[p1][p2]
            
    #     for i, buck in enumerate(a_rel):
    #         sen = adds[i]
    #         for p1 in range(seq_len):
    #             for p2 in range(seq_len):
    #                 for r in range(self.n_rels_add):
    #                     tup = (sen[p1], sen[p2], r)
    #                     score_rel[tup] = score_rel.get(tup, 0) + buck[p1][p2][r]


    #     bucket_size, seq_len = pos.shape

    #     for i in range(len(s_arc)):
    #         sen = pos[i]
    #         for p1 in range(seq_len):
    #             for p2 in range(seq_len):
    #                 tup = (sen[p1], sen[p2])
    #                 s_arc[i][p1][p2] += score_arc.get(tup, 0) * self.alpha

    #     for i in range(len(s_rel)):
    #         sen = pos[i]
    #         for p1 in range(seq_len):
    #             for p2 in range(seq_len):
    #                 for r in range(self.n_rels_add):
    #                     tup = (sen[p1], sen[p2], r)
    #                     s_rel[i][p1][p2][r] += score_rel.get(tup, 0) * self.alpha

    def modifyScore_2(self, adds, a_arc, a_rel, pos, s_arc, s_rel):
        n_pos = self.args.n_pos
        score_arc = torch.zeros([n_pos, n_pos], dtype=torch.float32).to(self.args.device)
        mapping = self.args.mapping

        batch_size, seq_len = adds.shape

        for b, _a_arc in enumerate(a_arc):
            for p1 in range(seq_len):
                for p2 in range(seq_len):
                    score_arc[adds[b][p1]][adds[b][p2]] = score_arc[adds[b][p1]][adds[b][p2]] + _a_arc[p1][p2]

        # print(pos.shape)
        
        score_rel = torch.zeros([n_pos, n_pos, self.args.n_rels_add], dtype=torch.float32).to(self.args.device)


        for b, _a_rel in enumerate(a_rel):
            for p1 in range(seq_len):
                for p2 in range(seq_len):
                    score_rel[adds[b][p1]][adds[b][p2]] = _a_rel[p1][p2] + score_rel[adds[b][p1]][adds[b][p2]]
        
        
        # Tmp = score_arc[pos, pos]
        # print(Tmp.shape)
        b, seq_len_tviet = pos.shape

        tmp = torch.zeros([b, seq_len_tviet, seq_len_tviet], dtype=torch.float32).to(self.args.device)

        for i in range(seq_len_tviet):
            for j in range(seq_len_tviet):
                tmp[:, i, j] = tmp[:, i, j] + score_arc[pos[:, i], pos[:, j]]

        tmp2 = torch.zeros([b, seq_len_tviet, seq_len_tviet, self.args.n_rels], dtype=torch.float32).to(self.args.device)

        for i in range(seq_len_tviet):
            for j in range(seq_len_tviet):
                for r in range(self.args.n_rels_add):
                    #if mapping[r] != self.args.bos_index:
                    if mapping[r] != 0:
                        tmp2[:, i, j, mapping[r]] = tmp2[:, i, j, mapping[r]] + score_rel[pos[:, i], pos[:, j], mapping[r]]

        return s_arc + torch.mul(tmp, self.alpha), s_rel + torch.mul(tmp2, self.alpha)
        
        #print(tmp.shape)

        pass

    def modifyScore_3(self, adds, a_arc, a_rel, pos, s_arc, s_rel):

        n_pos = self.args.n_pos
        bz, sl = adds.shape
        _add = adds.unsqueeze(1) #[bz, 1, sl]
        _add = _add.transpose(2, 1) * sl #[bz, sl, 1]
        _add = _add.repeat(1, 1, sl).reshape(bz, sl * sl) #[bz, sl, sl] -> [bz, sl * sl]
        _add = _add + adds.repeat(1, sl)
        _add = _add.reshape(bz, sl, sl)


        score_arc = torch.zeros([n_pos * n_pos], dtype=torch.float32).to(self.args.device)

        for it in range(n_pos * n_pos):
            mask = _add == it
            score_arc[it] = a_arc[mask].sum()

        score_arc = torch.sigmoid(score_arc)

        bz, sl = pos.shape
        _pos = pos.unsqueeze(1)
        _pos = _pos.transpose(2, 1) * sl
        _pos = _pos.repeat(1, 1, sl).reshape(bz, sl * sl)
        _pos = _pos + pos.repeat(1, sl)
        _pos = _pos.reshape(bz, sl, sl)

        for it in range(n_pos * n_pos):
            mask = _pos == it
            s_arc[mask] += score_arc[it] * self.alpha
        
        
        
        
