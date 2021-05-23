from utils import CoNLL
from tree import GraphSentence
from mod import modifyADP

x = CoNLL()

data = x.load('en_test.conllu')

s = dict()

sen = data[598]

a = GraphSentence(sen.value