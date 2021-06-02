#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train --feat bert --build -d 0 \
--train "code/data/vi_vtb-ud-train.conllu" \
--dev "code/data/vi_vtb-ud-dev.conllu" \
--test "code/data/vi_vtb-ud-test.conllu" \
--buckets 32 \
--punct \
--train_add "code/data/train_en.conllu" \
--dev_add "code/data/dev_en.conllu" \
--test_add "code/data/test_en.conllu" \
--n-embed 100 \
--embed "./glove.6B.100d.txt" \
--bert "vinai/phobert-base" \
--path "exp/test2" \
--batch-size 1000 \
--epochs 100000 \
--seed 1 \
--conf "./code/config.ini" \

