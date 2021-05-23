import sys
from utils import CoNLL
from tree import GraphSentence

instance = CoNLL()

data = instance.load(sys.argv[1])
f = open(sys.argv[2], "w")
sys.stdout = f

for sentence in data:
    tree = GraphSentence(sentence.values)

    order = tree.traverse(tree.root)
    
    #Tree traversal
    
    for i in range(1, len(sentence.values[0]) + 1):
        no = str(i)
        index = order[i] - 1
        #index = i - 1
        word = sentence.values[1][index]
        lemma = sentence.values[2][index]
        pos = sentence.values[3][index]
        xpos = sentence.values[4][index]
        feat = sentence.values[5][index]
        head = str(order.index(int(sentence.values[6][index])))
        #head = int(sentence.values[6][index])
        rel = sentence.values[7][index]
        dep = sentence.values[8][index]
        misc = sentence.values[9][index]
        if pos == 'ADV':
            pos = 'X'
        if pos == 'SYM':
            pos = 'PUNCT'
            rel = 'punct'
        if pos == 'PRON':
            pos = 'PROPN'

        print(no, word, lemma, pos, xpos, feat, head, rel, dep, misc, sep='\t')
    print()
    

f.close()


