from os import lseek
from code.utils.alg import kmeans

length = [2, 3, 4, 32, 32, 1, 24, 43, 22, 33, 55, 23, 23, 53, 44, 75, 33, 43, 22]
l = dict(zip(*kmeans(length, 5)))

trans = lambda x : list(map(lambda y : length[y], x))

y = list(map(lambda x : trans(x), l.values()))

k = list(map(lambda x: (min(x), max(x)), y))

print(k)