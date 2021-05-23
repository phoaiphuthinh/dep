# -*- coding: utf-8 -*-

from .parsers import BiaffineDependencyParser, EnsembleDependencyParser

__all__ = ['BiaffineDependencyParser',
            'EnsembleDependencyParser',
           'Parser',
           'EnsembleParser']

__version__ = '1.0.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser, EnsembleDependencyParser]}

PRETRAINED = {
    'biaffine-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.bert.zip',
}
