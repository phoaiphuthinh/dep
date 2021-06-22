#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train --feat char -d 0 \
--train "code/data/ptb/train.conllu" \
--dev "code/data/ptb/dev.conllu" \
--test "code/data/ptb/test.conllu" \
--buckets 32 \
--train_add "code/data/add/train.conllu" \
--n-embed 100 \
--embed "./glove.6B.100d.txt" \
--bert "vinai/phobert-base" \
--path "exp2/normal_2_thread_nobert_en" \
--batch-size 1000 \
--epochs 1000 \
--seed 1 \
--conf "./code/config.ini" \
