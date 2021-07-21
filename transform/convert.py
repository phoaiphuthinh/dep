import sys
import argparse
from utils import CoNLL
from tree import GraphSentence
from mod import modifyCase, modifyC, modifyMark

instance = CoNLL()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input file')
parser.add_argument('-o', '--output', help='Output file')
parser.add_argument('--case', help='Normalize the tree with case relation', action='store_true')
parser.add_argument('--cc', help='Normalize the tree with cc relation', action='store_true')
parser.add_argument('--mark', help='Normalize the tree with mark relation', action='store_true')
args = parser.parse_args()


maps = {'-LRB-' : 'LBKT', '-RRB-' : 'RBKT', '-LSB-' : 'LBKT', '-RSB-' : 'RBKT', '.' : 'SYM', ',' : 'SYM', ':' : 'SYM', "''" : 'SYM', '``' : 'SYM',
       'NNP' : 'Np', 'NNPS' : 'Np', 'JJ' : 'A', 'JJS' : 'A', 'JJR' : 'A', 'NN' : 'N', 'NNS' : 'N', 'POS' : 'E', 'WP' : 'N', 'WDT' : 'N', 'PRP' : 'P',
       'VB' : 'V', 'VBZ' : 'V', 'VBG' : 'V', 'VBD' : 'V', 'VBP' : 'V', 'RP': 'T', 'RB' : 'R', 'RBR' : 'R', 'RBS' : 'R', 'EX' : 'X', 'DT' : 'L', 'TO' : 'E',
       'CC' : 'C', 'CD' : 'M', 'IN' : 'E', 'UH' : 'X', 'WRB' : 'N', 'PRP$' : 'N', 'WP$' : 'N', 'MD' : 'R', 'LS' : 'X', 'VBN' : 'V', 'FW' : 'Y', 'PDT' : 'P'}

# (for French) mapping = {'NUM' : 'M', 'DET' : 'L', 'SYM' : 'SYM', 'PRON' : 'P', 'X' : 'X', 'INTJ' : 'T', 'CCONJ' : 'C', 'ADJ' : 'A', 'NOUN' : 'N', 'SCONJ' : 'C', 'PUNCT' : 'SYM', 'VERB' : 'V', 'ADV' : 'A', 'ADP' : 'E', 'PROPN' : 'P', 'AUX' : 'V'}


data = instance.load(args.input)
f = open(args.output, "w")
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
        
        head = str(order.index(int(sentence.values[6][index])))
        
        rel = sentence.values[7][index]
        
        if xpos in maps:
            xpos = maps[xpos]
        

        print(no, word, lemma, '_', xpos, '_', head, rel, '_', '_', sep='\t')
    
    print()
    

f.close()

if args.case or args.cc or args.mark:
    data = instance.load(args.output)
    f = open(args.output, "w")
    sys.stdout = f
    
    for sentence in data:
        x = sentence
        if args.case:
            x = modifyCase(x)
        if args.cc:
            x = modifyC(x)
        if args.mark:
            x = modifyMark(x)
        print(x)
    f.close()