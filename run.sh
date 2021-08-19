#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train -b -d 0  \
-c "code/config.ini"  \
-p "exp3/test_bertencoder_noModifyLabel_10k" \
-f bert \
--bert "vinai/phobert-base" \
--train "code/data/ptb/train.conllu" \
--dev "code/data/ptb/dev.conllu" \
--test "code/data/ptb/test.conllu" \
--train_add "code/data/add/new/en-vi-train.conllu" \
--batch-size 1000 \
--encoder "bert"

# python3.7 -m code.cmds.ensemble_dependency evaluate \
# --data "code/data/add_ptb/vi_dev.conllx" \
# -p "./exp3/test_trans_encode" \
# -d 0
