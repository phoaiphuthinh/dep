#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train --feat bert --build -d 0 \
--train "code/data/train_viet.conllu" \
--dev "code/data/dev_viet.conllu" \
--test "code/data/test_viet.conllu" \
--buckets 32 \
--punct \
--train_add "code/data/train_viet.conllu" \
--dev_add "code/data/dev_viet.conllu" \
--test_add "code/data/test_viet.conllu" \
--n-embed 100 \
--embed "./glove.6B.100d.txt" \
--bert "vinai/phobert-base" \
--path "exp/test3" \
--batch-size 1000 \
--epochs 100000 \
--seed 1 \
--conf "./code/config.ini" \

