from utils import CoNLL
from tree import GraphSentence

x = CoNLL()

data = x.load('en_gum-ud-test.conllu')


print(data[15].values)

x = GraphSentence(data[15].values)
l = x.traverse(x.root)
print(l)
