import sys
import random
from utils import CoNLL

instance = CoNLL()

data1 = instance.load(sys.argv[1])
data2 = instance.load(sys.argv[2])

random.shuffle(data1)
random.shuffle(data2)

f = open(sys.argv[3], "w")
sys.stdout = f

for x in data1[:4000]:
    print(x)

for y in data2[:6000]:
    print(y)
