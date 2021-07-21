class GraphSentence(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.root = Node()
        self.build(self.root)

    
    def build(self, pointer):
        if pointer is None:
            return
        for i, dep in enumerate(self.sentence[6]):
            if int(dep) == pointer.cur:
               
                chl = Node(self.sentence[3][i], pointer, int(self.sentence[0][i]))
                pointer.child.append(chl)
                relation = self.sentence[7][i]
                
                
                if i + 1 < pointer.cur:
                    pointer.pre.append(chl)
                else:
                    pointer.post.append(chl)
                
                
                self.build(chl)
             

    def traverse(self, node):
        if len(node.child) == 0:
            return [node.cur]
        s = []
        for i, child in enumerate(node.pre):
            ind = child.cur - 1
            if (self.sentence[2][ind] == 'ce' and self.sentence[7][ind] == 'det'):
                node.post = [child] + node.post
            else:
                s.extend(self.traverse(child))
        s.append(node.cur)
        
        for child in node.post:
            s.extend(self.traverse(child))
        
        return s


class Node(object):

    def __init__(self, POS=None, parent = None, cur=0):
        self.child = []
        self.POS = POS
        self.pre = []
        self.post = []
        self.other = []
        self.parent = parent
        self.cur = cur
