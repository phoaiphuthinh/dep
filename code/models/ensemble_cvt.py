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
#from code.models.dependency import BiaffineDependencyModel, AffineDependencyModel
from code.models.dependency2 import NoEmbeddingBiaffineDependencyModel, NoEmbeddingAffineDependencyModel
from code.models.embedding_layer import Embedding_Layer
from code.models.guessing_layer import Guessing_Layer


class EnsembleModel_CVT(nn.Module):


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

        self.embedding_layer = Embedding_Layer(n_words=n_words,
                                                n_feats=n_feats,
                                                n_rels=n_rels,
                                                feat=feat,
                                                n_embed=n_embed,
                                                n_feat_embed=n_feat_embed,
                                                n_char_embed=n_char_embed,
                                                bert=bert,
                                                n_feats_add=n_feats_add,
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
                                                unk_index=unk_index)
        
        self.embedding_layer_add = Embedding_Layer(n_words=n_feats_add,
                                                n_feats=n_feats,
                                                n_rels=n_rels,
                                                feat=feat,
                                                n_embed=n_embed,
                                                n_feat_embed=n_feat_embed,
                                                n_char_embed=n_char_embed,
                                                bert=bert,
                                                n_feats_add=n_feats_add,
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
                                                unk_index=unk_index)
        
        # self.embedding_layer_add = Embedding_Layer(n_words=n_feats_add,
        #                                         n_feats=n_feats,
        #                                         n_rels=n_rels,
        #                                         feat=feat,
        #                                         n_embed=n_embed,
        #                                         n_feat_embed=n_feat_embed,
        #                                         n_char_embed=n_char_embed,
        #                                         bert=bert,
        #                                         n_bert_layers=n_bert_layers,
        #                                         mix_dropout=mix_dropout,
        #                                         embed_dropout=embed_dropout,
        #                                         n_lstm_hidden=n_lstm_hidden,
        #                                         n_lstm_layers=n_lstm_layers,
        #                                         lstm_dropout=lstm_dropout,
        #                                         n_mlp_arc=n_mlp_arc,
        #                                         n_mlp_rel=n_mlp_rel,
        #                                         mlp_dropout=mlp_dropout,
        #                                         feat_pad_index=feat_pad_index,
        #                                         pad_index=pad_index,
        #                                         unk_index=unk_index)

        # self.origin = NoEmbeddingBiaffineDependencyModel(n_words=n_words,
        #                                         n_feats=n_feats,
        #                                         n_rels=n_rels,
        #                                         feat=feat,
        #                                         n_embed=n_embed,
        #                                         n_feat_embed=n_feat_embed,
        #                                         n_char_embed=n_char_embed,
        #                                         bert=bert,
        #                                         n_bert_layers=n_bert_layers,
        #                                         mix_dropout=mix_dropout,
        #                                         embed_dropout=embed_dropout,
        #                                         n_lstm_hidden=n_lstm_hidden,
        #                                         n_lstm_layers=n_lstm_layers,
        #                                         lstm_dropout=lstm_dropout,
        #                                         n_mlp_arc=n_mlp_arc,
        #                                         n_mlp_rel=n_mlp_rel,
        #                                         mlp_dropout=mlp_dropout,
        #                                         feat_pad_index=feat_pad_index,
        #                                         pad_index=pad_index,
        #                                         unk_index=unk_index) #More argument

        self.origin = NoEmbeddingAffineDependencyModel(n_feats_add=n_feats_add,
                                                n_rels_add=n_rels,
                                                embed_dropout=embed_dropout,
                                                n_lstm_hidden=n_lstm_hidden,
                                                n_lstm_layers=n_lstm_layers,
                                                lstm_dropout=lstm_dropout,
                                                n_mlp_arc=n_mlp_arc,
                                                n_mlp_rel=n_mlp_rel,
                                                mlp_dropout=mlp_dropout,
                                                pad_index_add=pad_index_add,
                                                unk_index_add=unk_index_add) #More

        self.addition = NoEmbeddingAffineDependencyModel(n_feats_add=n_feats_add,
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
        
        self.guessing_layer = Guessing_Layer(n_in=n_lstm_hidden*2, n_out=n_embed, pad_index=pad_index, dropout=mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_unsupervised = nn.MSELoss(reduction='mean')
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.alpha = alpha
        self.n_rels_add = n_rels
        #self.redirect


    def load_pretrained(self, embed=None):
        # if embed is not None:
        #     self.embedding_layer_add = self.embedding_layer_add.load_pretrained(embed)
        return self
    
    def word_guessing_tviet_forward(self, words, feats, pos):
        self._freeze_model()
        self.embedding_layer_add._freeze()
        self.addition._freeze()
        self.guessing_layer._unfreeze()

        word_embed_tviet, feat_embed_tviet, mask_tviet, seq_len_tviet = self.embedding_layer(words, feats)

        word_embed_tanh, _, mask_tanh, seq_len_tanh = self.embedding_layer_add(pos)
        a_arc, a_rel, lstm = self.addition(word_embed_tanh, mask=mask_tanh, seq_len=seq_len_tanh, cvt=True)
        guessed_ = self.guessing_layer(lstm, mask=mask_tanh)
        
        # with open("test.txt", "a+") as f:
        #     print(guessed_, file=f)
        #     print(guessed_.shape, file=f)
        #     print(word_embed_tviet, file=f)
        #     print(word_embed_tviet.shape, file=f)

        loss = self.criterion_unsupervised(guessed_, word_embed_tviet)
        return loss


    def forward(self, words, feats, adds=None, pos=None, cvt=False, trainpos=False):
        
        if trainpos:
            return self.word_guessing_tviet_forward(words, feats, pos)
        
        self._unfreeze_model()
        self.embedding_layer_add._unfreeze()
        self.addition._unfreeze()
        self.guessing_layer._freeze()

        if not cvt:
            
            self._unfreeze_model()

            word_embed_tviet, feat_embed_tviet, mask_tviet, seq_len_tviet = self.embedding_layer(words, feats)
            
            #s_arc, s_rel = self.origin(word_embed_tviet, feat_embed_tviet, mask=mask_tviet, seq_len=seq_len_tviet) #biaffine
            s_arc, s_rel = self.origin(word_embed_tviet, mask=mask_tviet, seq_len=seq_len_tviet) #affine

            if adds is not None:
                
                word_embed_tanh, _, mask_tanh, seq_len_tanh = self.embedding_layer_add(adds)

                a_arc, a_rel, lstm = self.addition(word_embed_tanh, mask=mask_tanh, seq_len=seq_len_tanh, cvt=True)

                guessed_ = self.guessing_layer(lstm, mask=mask_tanh)

                # guess_arc, guess_rel = self.origin(guessed_, mask=mask_tanh, seq_len=seq_len_tanh)
                # guess_arc = guess_arc.detach()
                # guess_rel = guess_rel.detach()

                #a_arc = a_arc
                #a_rel = a_rel + 0.5 * guess_rel
            
                self.modifyScore(adds, a_arc, pos, s_arc)
                
                return s_arc, s_rel, a_arc, a_rel


            word_embed_tanh, _, mask_tanh, seq_len_tanh = self.embedding_layer_add(pos)

            a_arc, a_rel, lstm = self.addition(word_embed_tanh, mask=mask_tanh, seq_len=seq_len_tanh, cvt=True)

            # guessed_ = self.guessing_layer(lstm, mask=mask_tanh)

            # guess_arc, guess_rel = self.origin(guessed_, mask=mask_tanh, seq_len=seq_len_tanh)
            # guess_arc = guess_arc.detach()
            # guess_rel = guess_rel.detach()

            # a_arc = a_arc * 0.5 + guess_arc

            self.modifyScore(pos, a_arc, pos, s_arc)

            return s_arc, s_rel

        else:
            self._freeze_model()
            self.guessing_layer._unfreeze()

            word_embed_tanh, _, mask_tanh, seq_len_tanh = self.embedding_layer_add(adds)

            a_arc, a_rel, lstm = self.addition(word_embed_tanh, mask=mask_tanh, seq_len=seq_len_tanh, cvt=True)

            guessed_ = self.guessing_layer(lstm, mask=mask_tanh)

            s_arc, s_rel = self.origin(guessed_, mask=mask_tanh, seq_len=seq_len_tanh)

            s_arc = s_arc.detach()
            s_rel = s_rel.detach()

            loss = (self.criterion_unsupervised(a_arc, s_arc))
            
            return loss
        #pass

    def _freeze_model(self):
        # self.encoder.eval()
        self.origin._freeze()
        self.embedding_layer._freeze()
        # for params in self.encoder.parameters():
        #     params.requires_grad = False
        # for params in self.embedding_layer.parameters():
        #     params.requires_grad = False

    def _unfreeze_model(self):
        # self.encoder.train()
        #self.origin.train()
        self.embedding_layer._unfreeze()
        self.origin._unfreeze()
        # for params in self.encoder.parameters():
        #     params.requires_grad = True
        # for params in self.origin.parameters():
        #     params.requires_grad = True
        # for params in self.embedding_layer.parameters():
        #     params.requires_grad = True

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
            return self.origin.loss(s_arc, s_rel, arcs, rels, mask) + self.addition.loss(a_arc, a_rel, arcs_add, rels_add, mask_add) * self.alpha

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

    def modifyScore(self, adds, a_arc, pos, s_arc):

        n_pos = self.args.n_pos
        bz, sl = adds.shape
        _add = adds.unsqueeze(1) #[bz, 1, sl]
        _add = _add.transpose(2, 1) * n_pos #[bz, sl, 1]
        _add = _add.repeat(1, 1, sl).reshape(bz, sl * sl) #[bz, sl, sl] -> [bz, sl * sl]
        _add = _add + adds.repeat(1, sl)
        _add = _add.reshape(bz, sl, sl)

        assert torch.all(_add < n_pos * n_pos)

        score_arc = torch.zeros([n_pos * n_pos], dtype=torch.float32).to(self.args.device)

        for it in range(n_pos * n_pos):
            mask = _add == it
            score_arc[it] = a_arc[mask].sum()

        score_arc = torch.sigmoid(score_arc)

        bz, sl = pos.shape
        _pos = pos.unsqueeze(1)
        _pos = _pos.transpose(2, 1) * n_pos
        _pos = _pos.repeat(1, 1, sl).reshape(bz, sl * sl)
        _pos = _pos + pos.repeat(1, sl)
        _pos = _pos.reshape(bz, sl, sl)

        assert torch.all(_pos < n_pos * n_pos)

        for it in range(n_pos * n_pos):
            mask = _pos == it
            s_arc[mask] += score_arc[it] * self.alpha
        