# Syntactic transformation

## Usage
This allows users to do syntactic transformation from data in CoNLL-U format. 

If you want to map the tags from source language to target language, define the a dictionary in  `convert.py` 

```python
maps = {'JJS' : 'A', 'NN' : 'N', ...}
```

Define rules to reorder the words in sentences, the function  `traverse` in `tree.py`    

```python
def traverse(self, node):
    ....
        for i, child in enumerate(node.pre):
            <Modify here>
        s.append(node.cur)
        for i, child in enumerate(node.post):
            <Modify here>
        ....
```

If you want to modify arcs so that they are in the same annotation style with target language treebank, write the function in `mod.py` and import them. Add the option to the `argparse` to use it when running:
```python
parser.add_argument('--case', help='Normalize the tree with case relation', action='store_true')
```
Finally, run the command line:
```sh
$ python convert.py -i <input_file> -o <output_name> [--case] [--cc] [--mark] [--other_options]
```

For example:
```sh
$ python convert.py -i en_gum-ud-train.connlu -o transformed.conllu --case
```
One sentence after converted:
```python
1	Next	next	_	R	_	4	advmod	_	_
2	,	,	_	SYM	_	1	punct	_	_
3	we	we	_	P	_	4	nsubj	_	_
4	present	present	_	V	_	0	root	_	_
5	the	the	_	L	_	6	det	_	_
6	results	result	_	N	_	4	obj	_	_
7	of	of	_	E	_	6	case	_	_
8	our	our	_	N	_	9	nmod:poss	_	_
9	work	work	_	N	_	7	nmod	_	_
10	in	in	_	E	_	4	case	_	_
11	two	two	_	M	_	12	nummod	_	_
12	sections	section	_	N	_	10	obl	_	_
13	.	.	_	SYM	_	4	punct	_	_
```

