from utils import CoNLL
from tree import GraphSentence
from mod import modifyADP
import matplotlib.pyplot as plt
x = CoNLL()

data = x.load('en_train.conllu')

s = dict()
m = max(list(map(lambda x : len(x.values[0]), data)))
for x in data:
    l = len(x.values[0])
    s[l] = s.get(l, 0) + 1

values = []
names = []
for i in range(m):
    values.append(s.get(i+1, 0))
    names.append(i+1)
    
plt.bar(names, values)
plt.show()

print(values)