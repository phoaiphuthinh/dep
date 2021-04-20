WALS = {('NOUN', 'ADP') : -2, ('NOUN', 'ADJ') : 2, ('NOUN', 'DET') : -2, ('NOUN', 'NUM') : -2}

statPOS = {('NOUN', 'ADP') : -1, ('NOUN', 'DET') : -1, ('PROPN', 'ADP') : -1, ('NOUN', 'VERB') : 1, 
            ('NOUN', 'AUX') : -1, ('PROPN', 'CCONJ') : -1, ('ADJ', 'VERB') : 1, ('ADJ', 'ADJ') : 1,
            ('VERB', 'AUX') : -1, ('NOUN', 'ADJ') : 1}

statHead = {'nsub' : -1, 'case' : -1, 'cop' : -1, 'advcl' : -1, 'csubj' : -1, 'aux' : -1, 
            'admod' : 1, 'obj' : 1, 'ccomp' : 1, 'parataxis' : 1, 'conj' : 1, 'appos' : 1, 'iobj' : 1}

class GraphSentence(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.root = Node()
        self.build(self.root)

    
    def build(self, pointer):
        if pointer is None:
            return
        #print(pointer.cur)
        for i, dep in enumerate(self.sentence[6]):
            if int(dep) == pointer.cur:
                chl = Node(self.sentence[3][i], pointer, i + 1)
                pointer.child.append(chl)
                relation = self.sentence[7][i]
                pointer.rel.append(relation)
                

                tup = (pointer.POS, pointer.child[-1].POS)
                wals = WALS.get(tup, 0)
                pos = statPOS.get(tup, 0)
                head = statHead.get(relation, 0)
                score = wals + pos + head

                if score < 0:
                    pointer.pre.append(chl)
                elif score > 0:
                    pointer.post.append(chl)
                else:
                    pointer.other.append(chl)
                self.build(chl)

    def traverse(self, node):
        if len(node.child) == 0:
            return [node.cur]
        s = []
        x = [c for c in node.other if c.cur < node.cur]
        pre = node.pre + x
        pre.sort(key=lambda x : x.cur)
        print([w.cur for w in pre])
        for c in pre:
            s.extend(self.traverse(c))
        s.append(node.cur)

        x = [c for c in node.other if c.cur > node.cur]
        post = node.post + x
        post.sort(key=lambda x : x.cur)
        print([w.cur for w in post])
        for c in post:
            s.extend(self.traverse(c))

        return s


class Node(object):

    def __init__(self, POS=None, parent = None, cur=0):
        self.child = []
        self.rel = []
        self.POS = POS
        self.pre = []
        self.post = []
        self.other = []
        self.parent = parent
        self.cur = cur