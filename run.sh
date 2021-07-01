#! /bin/bash

python3.7 -m code.cmds.ensemble_crf2o train --feat bert -d 0 \
-b \
--mbr \
--train "code/data/ptb/train.conllu" \
--dev "code/data/ptb/dev.conllu" \
--test "code/data/ptb/test.conllu" \
--buckets 32 \
--train_add "code/data/en_vi.conllu" \
--n-embed 100 \
--embed "./glove.6B.100d.txt" \
--bert "vinai/phobert-base" \
--path "exp2/crf_ensemble_test2" \
--batch-size 1000 \
--epochs 1000 \
--seed 1 \
--conf "./code/config.ini" \
