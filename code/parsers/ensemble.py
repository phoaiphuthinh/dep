from code.parsers.ensembleparser import EnsembleParser
import os

import torch
import torch.nn as nn
from code.models import (EnsembleModel)
from code.parsers.parser import Parser
from code.utils import Config, Dataset, Embedding
from code.utils.common import bos, pad, unk
from code.utils.field import Field, SubwordField
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

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.origin.FORM
        else:
            self.WORD, self.FEAT = self.origin.FORM, self.origin.CPOS
        self.ARC, self.REL = self.origin.HEAD, self.origin.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(self.args.device)

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
            return parser

        logger.info("Building the fields")
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
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            origin = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            origin = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(origin, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(train)
        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index,
        })
        logger.info(f"{origin}")


        logger.info("Building the fields")
        POS = Field('tags', bos=bos)
        ARC_ADD = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL_ADD = Field('rels', bos=bos)
        addition = CoNLL(CPOS=POS, HEAD=ARC_ADD, DEPREL=REL_ADD)

        train = Dataset(addition, args.train)
        
        POS.build(train)
        REL_ADD.build(train)
        
        # args.update({
        #     'n_words': WORD.vocab.n_init,
        #     'n_feats': len(FEAT.vocab),
        #     'n_rels': len(REL.vocab),
        #     'pad_index': WORD.pad_index,
        #     'unk_index': WORD.unk_index,
        #     'bos_index': WORD.bos_index,
        #     'feat_pad_index': FEAT.pad_index,
        # })
        logger.info(f"{addition}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed).to(args.device)
        logger.info(f"{model}\n")

        optimizer = Adam(model.parameters(), **optimizer_args)
        scheduler = ExponentialLR(optimizer, **scheduler_args)

        return cls(args, model, origin, addition, optimizer, scheduler)