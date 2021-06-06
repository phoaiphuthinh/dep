def modifyOf(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    for i in range(len(sentence.values[0])):
        word = sentence.values[2][i]
        pos = sentence.values[3][i]
        if word == 'of' and pos == 'ADP':
            par = int(sentence.values[6][i])
            grp = int(sentence.values[6][par - 1])
            sentence.values[6][i] = grp
            sentence.values[6][par - 1] = i + 1
            sentence.values[7][i] = 'nmod'
            sentence.values[7][par - 1] = 'pob'
    return sentence

def modifyADP(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
     
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        if pos == 'E':
            rel = sentence.values[7][i]
            par = int(sentence.values[6][i])
            
            relp = sentence.values[7][par - 1]
            grp = int(sentence.values[6][par - 1])
            
            if rel == 'case' and relp == 'nmod':
                sentence.values[6][i] = grp
                sentence.values[6][par - 1] = i + 1
                sentence.values[7][i] = 'nmod'
                sentence.values[7][par - 1] = 'pob'
    return sentence 

def modifyCompoundNoun(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        rel = sentence.values[7][i]
        
        if rel == 'compound' and (pos == 'N' or pos == 'P'):
            sentence.values[7][i] = 'nmod'
        if pos == 'E' and rel == 'compound':
            par = int(sentence.values[6][i])
            posp = sentence.values[4][par - 1]
            if posp == 'V':
                sentence.values[7][i] = 'vmod'
    
    return sentence


def modifyPoss(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        lemma = sentence.values[2][i]
        
        if lemma == '\'s':
            par = int(sentence.values[6][i])
            if sentence.values[7][par - 1] == 'nmod:poss':
                
                sentence.values[7][par - 1] = 'nmod'
                sentence.values[1][i] = 'of'
            
    return sentence


def modifyCase(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        rel = sentence.values[7][i]
        
        if rel == 'case' and pos == 'E':
            par = int(sentence.values[6][i])
            grp = int(sentence.values[6][par - 1])
            
            gpos = sentence.values[4][grp - 1]
            
            sentence.values[6][i] = grp
            sentence.values[6][par - 1] = i + 1
            sentence.values[7][par - 1] = 'pob'
            
            if gpos == 'N' or gpos == 'P':
                sentence.values[7][i] = 'nmod'
            elif gpos == 'A':
                sentence.values[7][i] = 'amod'
            elif gpos == 'V':
                sentence.values[7][i] = 'vmod'
            else:
                sentence.values[7][i] = 'x' #If not match at all
    return sentence

def modifyC(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        rel = sentence.values[7][i]
        
        if rel == 'cc' and pos == 'C':
            par = int(sentence.values[6][i])
            grp = int(sentence.values[6][par - 1])
            
            gpos = sentence.values[4][grp - 1]
            
            sentence.values[6][i] = grp
            sentence.values[6][par - 1] = i + 1
            sentence.values[7][par - 1] = 'conj'
            sentence.values[7][i] = 'coord'
    return sentence

def modifyMark(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        rel = sentence.values[7][i]
        
        if rel == 'mark' and pos == 'E':
            par = int(sentence.values[6][i])
            grp = int(sentence.values[6][par - 1])
            
            gpos = sentence.values[4][grp - 1]
            
            sentence.values[6][i] = grp
            sentence.values[6][par - 1] = i + 1
            sentence.values[7][par - 1] = 'pob'
            sentence.values[7][i] = 'loc'
    return sentence