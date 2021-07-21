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
    return sentence

def modifyMark(sentence):
    for i in range(len(sentence.values)):
        sentence.values[i] = list(sentence.values[i])
    
    for i in range(len(sentence.values[0])):
        pos = sentence.values[4][i]
        rel = sentence.values[7][i]
        
        if rel == 'mark' and (pos == 'E' or pos == 'C'):
            par = int(sentence.values[6][i])
            grp = int(sentence.values[6][par - 1])
            
            gpos = sentence.values[4][grp - 1]
            
            sentence.values[6][i] = grp
            sentence.values[6][par - 1] = i + 1
    return sentence