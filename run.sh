#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train -b -d 0 \
-c "code/config.ini"  \
-p "rate_train_pos_originaltreebank_punct/test_seperate_train" \
-f bert \
--punct \
--tree \
--bert "vinai/phobert-base" \
--train "code/data/org/VnDTv1.1-predicted-POS-tags-train.conll" \
--dev "code/data/org/VnDTv1.1-predicted-POS-tags-dev.conll" \
--test "code/data/org/VnDTv1.1-predicted-POS-tags-test.conll" \
--train_add "code/data/org/envi-train.conllu" \
--dev_add "code/data/org/envi-dev.conllu" \
--test_add "code/data/org/envi-test.conllu" \
--batch-size 1000 \
--epochs 50 \
--epochs_add 300 \
--encoder "bert"

# python3.7 -m code.cmds.ensemble_dependency evaluate \
# --data "code/data/add_ptb/vi_dev.conllx" \
# -p "./exp3/test_trans_encode" \
# -d 0
