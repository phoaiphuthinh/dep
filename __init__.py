# -*- coding: utf-8 -*-

from .parsers import (BiaffineDependencyParser)

__all__ = ['BiaffineDependencyParser'
           'Parser']

__version__ = '1.0.0'

PARSER = {parser.NAME: parser for parser in [BiaffineDependencyParser]}

PRETRAINED = {
    'biaffine-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.bert.zip',
}
