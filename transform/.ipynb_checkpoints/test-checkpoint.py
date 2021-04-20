from utils import CoNLL
from tree import GraphSentence

def merge(d, newd):
    for key, val in newd.items():
        d[key] = d.get(key, 0) + val 
    return d

x = CoNLL()

data = x.load('vi_vtb-ud-test.conllu')

left = {}
right = {}
headl = {}
headr = {}

for sent in data:
    x = GraphSentence(sent.values)
    left = merge(left, x.left)
    right = merge(right, x.right)
    headl = merge(headl, x.headl)
    headr = merge(headr, x.headr)
print(sorted(left.items(), key=lambda x : x[1]))
print(sorted(right.items(), key=lambda x : x[1]))
print(sorted(headl.items(), key=lambda x : x[1]))
print(sorted(headr.items(), key=lambda x : x[1]))