# -*- coding: utf-8 -*-

import argparse


from code import EnsembleDependencyParser

from code.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivise the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=EnsembleDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'bert'], help='choices of additional features')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='code/data/ptb/train.conllu', help='path to train file')
    subparser.add_argument('--dev', default='code/data/ptb/dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default='code/data/ptb/test.conllu', help='path to test file')

    subparser.add_argument('--embed', help='path to pretrained embeddings')
    
    subparser.add_argument('--train_add', default='code/data/add/train.conllu', help='Path to train file')
    subparser.add_argument('--dev_add', default='code/data/add/train.conllu', help='Path to dev file')
    subparser.add_argument('--test_add', default='code/data/add/train.conllu', help='Path to train file')

    subparser.add_argument('--use_cpos', action='store_true')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    subparser.add_argument('--encoder', default='lstm')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='code/data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--log_file', default='code/data/ptb/log_acc.txt')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='code/data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
