#! /bin/bash

python3.7 -m code.cmds.biaffine_dependency train --feat char -d 0 \
-b \
--train "code/data/ptb/train.conllu" \
--dev "code/data/ptb/dev.conllu" \
--test "code/data/ptb/test.conllu" \
--buckets 32 \
--n-embed 100 \
--embed "./glove.6B.100d.txt" \
--bert "vinai/phobert-base" \
--path "exp2/refactor_code" \
--batch-size 1000 \
--epochs 1000 \
--seed 1 \
--conf "./code/config.ini" \
