from torch.serialization import load
from code.parsers.ensembleparser import EnsembleParser
import os
import re
from collections import Counter

import torch
import torch.nn as nn
from code.models import (EnsembleModel)
from code.parsers.parser import Parser
from code.utils import Config, Dataset, Embedding
from code.utils.common import bos, pad, unk
from code.utils.field import RawField, Field, SubwordField
from code.utils.vocab import Vocab
from code.utils.fn import ispunct
from code.utils.logging import get_logger, progress_bar
from code.utils.metric import AttachmentMetric
from code.utils.transform import CoNLL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger(__name__)

class EnsembleDependencyParser(EnsembleParser):

    NAME = 'ensemble-dependency'
    MODEL = EnsembleModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        if self.args.encoder != 'bert':
            if self.args.feat in ('char', 'bert'):
                self.WORD, self.TEXT, self.FEAT = self.origin.FORM
            else:
                self.WORD, self.TEXT = self.origin.FORM
                self.FEAT = self.origin.POS
                if args.use_cpos:
                    self.WORD, self.TEXT = self.origin.FORM
                    self.FEAT = self.origin.CPOS

        else:
            self.WORD, self.TEXT = self.origin.FORM
        self.ARC, self.REL = self.origin.HEAD, self.origin.DEPREL
            # self.puncts = torch.tensor([i
            #                             for s, i in self.WORD.vocab.stoi.items()
            #                             if ispunct(s)]).to(self.args.device)

        self.POS = self.addition.POS
        if self.args.use_cpos:
            self.POS = self.addition.CPOS
        self.ARC_ADD, self.REL_ADD = self.addition.HEAD, self.addition.DEPREL


    def train(self, train, dev, test, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def _train_2_time(self, loader, source_train=False):
        self.model.train()

        pad_index = self.args.feat_pad_index_add if source_train else self.args.pad_index

        bar, metric = progress_bar(loader), AttachmentMetric()

        for it in bar:
            if source_train:
                texts, words, arcs, rels = it
                feats = None
                pos = None
            else:
                if self.args.encoder != 'bert':
                    if self.args.feat in ('char', 'bert'):
                        words, texts, feats, pos, arcs, rels = it
                    else:
                        words, texts, feats, arcs, rels = it
                        pos = feats
                else:
                    words, texts, pos, arcs, rels = it
                    feats = None

            self.optimizer.zero_grad()
            word_mask = words.ne(pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0

            s_arc, s_rel = self.model(words=words, feats=feats, pos=pos, source_train=source_train)

            loss = self.model.loss_2_time(s_arc=s_arc, s_rel=s_rel, arcs=arcs, rels=rels, mask=mask, partial=self.args.partial)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)

            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in texts for w in s]))

            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")
        

    def _train(self, loader, loader_add):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()
        bar_add = iter(loader_add)
        
        for it in bar:
            if self.args.encoder != 'bert':
                if self.args.feat in ('char', 'bert'):
                    words, texts, feats, pos, arcs, rels = it
                else:
                    words, texts, feats, arcs, rels = it
                    pos = feats
            else:
                words, texts, pos, arcs, rels = it
                feats = None
            self.optimizer.zero_grad()
            it_nx = next(bar_add, None)
            if (it_nx is None):
                bar_add = iter(loader_add)
                text_add, words_add, arcs_add, rels_add = next(bar_add)
            else:
                text_add, words_add, arcs_add, rels_add = it_nx
            #print(words_add)
            #print(pos)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask_add = words_add.ne(self.POS.pad_index)

            # ignore the first token of each sentence
            mask[:, 0] = 0
            mask_add[:, 0] = 0
            s_arc, s_rel, a_arc, a_rel = self.model(words, feats, words_add, pos)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, a_arc, a_rel, arcs_add, rels_add, mask_add, self.args.partial)
            # print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            # if not self.args.punct:
            #     mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)

            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in texts for w in s]))

            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader, source_train=False):
        self.model.eval()

        pad_index = self.args.feat_pad_index_add if source_train else self.args.pad_index

        total_loss, metric = 0, AttachmentMetric()

        for it in loader:
            if source_train:
                texts, words, arcs, rels = it
                feats = None
                pos = None
            else:
                if self.args.encoder != 'bert':
                    if self.args.feat in ('char', 'bert'):
                        words, texts, feats, pos, arcs, rels = it
                    else:
                        words, texts, feats, arcs, rels = it
                        pos = feats
                else:
                    words, texts, pos, arcs, rels = it
                    feats = None
            word_mask = words.ne(pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, pos=pos, source_train=source_train)
            loss = self.model.loss_2_time(s_arc=s_arc, s_rel=s_rel, arcs=arcs, rels=rels, mask=mask, partial=self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified

            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in texts for w in s]))

            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _evaluate_print(self, loader):
        self.model.eval()

        file = self.args.log_file
        f = open(file, 'w')
        
        total_loss, metric = 0, AttachmentMetric()
        for it in loader:
            if self.args.encoder != 'bert':
                if self.args.feat in ('char', 'bert'):
                    words, texts, feats, pos, arcs, rels = it
                else:
                    words, texts, feats, arcs, rels = it
                    pos = feats
            else:
                words, texts, pos, arcs, rels = it
                feats = None
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, pos=pos)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)

            # ignore all punctuation if not specified
            # if not self.args.punct:
            #     mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)

            if not self.args.punct:
                mask.masked_scatter_(mask, ~mask.new_tensor([ispunct(w) for s in texts for w in s]))

            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)

            bz, sl = mask.shape
            #For each sentence -> print the accuracy
            for _ in range(bz):
                arc_prd, rel_prd = arc_preds[_, :], rel_preds[_, :]
                arc_gold, rel_gold = arcs[_, :], rels[_, :]
                _mask = mask[_, :]
                arc_mask = arc_prd.eq(arc_gold) & _mask
                rel_mask = rel_prd.eq(rel_gold) & arc_mask
                arc_mask_seq, rel_mask_seq = arc_mask[_mask], rel_mask[_mask]
                total = len(arc_mask_seq)
                correct_arcs = arc_mask_seq.sum().item()
                correct_rels = rel_mask_seq.sum().item()
                uas = correct_arcs / (total + 1e-12)
                las = correct_rels / (total + 1e-12)
                f.write(f"UAS: {uas:6.2%} LAS: {las:6.2%} sentence_length: {total}\n")
        f.close()
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        full_probs = []
        for it in loader:
            if self.args.encoder != 'bert':
                if self.args.feat in ('char', 'bert'):
                    words, texts, feats, pos, arcs, rels = it
                else:
                    words, texts, feats, arcs, rels = it
                    pos = feats
            else:
                words, texts, pos, arcs, rels = it
                feats = None
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats, pos=pos)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, self.args.tree, self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def getTagSet(cls, f_name, bos):
        if os.path.isfile(f_name):
            with open(f_name, 'r') as f:
                data = f.read()
                tag_set = re.split('\n|\r\n|\r', data)
                c = Counter(tag_set)
                vocab = Vocab(c, 1, [bos], None)
                return vocab
        return None

    @classmethod
    def build(cls, path,
              optimizer_args={'lr': 2e-3, 'betas': (.9, .9), 'eps': 1e-12},
              scheduler_args={'gamma': .75**(1/5000)},
              min_freq=2,
              fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            optimizer_args (dict):
                Arguments for creating an optimizer.
            scheduler_args (dict):
                Arguments for creating a scheduler.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            parser.optimizer = Adam(parser.model.parameters(), **optimizer_args)
            parser.scheduler = ExponentialLR(parser.optimizer, **scheduler_args)
            return parser

        logger.info("Building the fields")

        FEAT = None
        TAG = None

        if args.encoder == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=False)
            WORD = SubwordField('words',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            WORD.vocab = tokenizer.get_vocab()

        else:

            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if args.feat == 'char':
                FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
            elif args.feat == 'bert':
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.bert)
                FEAT = SubwordField('bert',
                                    pad=tokenizer.pad_token,
                                    unk=tokenizer.unk_token,
                                    bos=tokenizer.bos_token or tokenizer.cls_token,
                                    fix_len=args.fix_len,
                                    tokenize=tokenizer.tokenize)
                FEAT.vocab = tokenizer.get_vocab()
            else:
                FEAT = Field('tags', bos=bos)

        TEXT = RawField('texts')
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        TAG = Field('pos', bos=bos, pad=pad)

        if args.encoder != 'bert':
            if args.feat in ('char', 'bert'):
                if args.use_cpos:
                    origin = CoNLL(FORM=(WORD, TEXT, FEAT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
                else:
                    origin = CoNLL(FORM=(WORD, TEXT, FEAT), POS=TAG, HEAD=ARC, DEPREL=REL)
            else:
                if args.use_cpos:
                    origin = CoNLL(FORM=(WORD, TEXT), CPOS=FEAT, HEAD=ARC, DEPREL=REL)
                else:
                    origin = CoNLL(FORM=(WORD, TEXT), POS=FEAT, HEAD=ARC, DEPREL=REL)
        else:
            if args.use_cpos:
                origin = CoNLL(FORM=(WORD, TEXT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
            else:
                origin = CoNLL(FORM=(WORD, TEXT), POS=TAG, HEAD=ARC, DEPREL=REL)

        tag_set = EnsembleDependencyParser.getTagSet(args.tag_set_path, bos)
        if tag_set != None:
            TAG.vocab = tag_set

        train = Dataset(origin, args.train)

        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
            FEAT.build(train)
        else:
            WORD.build(train)
        REL.build(train)
        if args.feat in ('char', 'bert'):
            TAG.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab) if args.encoder != 'bert' else 100,
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index if args.encoder != 'bert' else 0,
        })
        logger.info(f"{origin}")

        logger.info("Building the fields")
        TEXT_ADD = RawField('texts')
        POS = Field('tags', bos=bos, pad=pad)
        ARC_ADD = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL_ADD = Field('rels', bos=bos)

        if args.use_cpos:
            addition = CoNLL(FORM=TEXT_ADD, CPOS=POS, HEAD=ARC_ADD, DEPREL=REL_ADD)
        else:
            addition = CoNLL(FORM=TEXT_ADD, POS=POS, HEAD=ARC_ADD, DEPREL=REL_ADD)

        train_add = Dataset(addition, args.train_add)

        if not args.use_cpos:
            if args.feat in ('char', 'bert'):
                POS.vocab = TAG.vocab
            else:
                POS.vocab = FEAT.vocab

        REL_ADD.vocab = REL.vocab

        if tag_set != None:
            POS.vocab = tag_set

        # for (k, v) in POS.vocab.stoi.items():
        #     print(k, v)

        POS.build(train_add)
        REL_ADD.build(train_add)
        TAG.vocab = POS.vocab
        print(TAG.vocab.stoi)
        print(POS.vocab.stoi)
        
        args.update({
            'n_feats_add': len(POS.vocab),
            'n_rels_add': len(REL_ADD.vocab),
            'feat_pad_index_add': POS.pad_index,
            'n_pos' : len(POS.vocab)
        })
        logger.info(f"{addition}")


        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device) if args.encoder != 'bert' else cls.MODEL(**args).load_pretrained().to(args.device)
        logger.info(f"{model}\n")

        print(model)

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, origin, addition, optimizer, scheduler)