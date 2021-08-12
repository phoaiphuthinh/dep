#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train -b -d 0  \
-c config.ini  \
-p "exp3/test_trans_encode_bilstm" \
-f bert \
--bert "vinai/phobert-base" \
--epochs 1000 \
--train "code/data/add_ptb/vi_train.conllx" \
--dev "code/data/add_ptb/vi_dev.conllx" \
--test "code/data/add_ptb/vi_test.conllx" \
--train_add "code/data/add_ptb/en_full.conllx" \
--batch-size 5000 \
--tag_set_path "code/data/pos_set.txt" \
--use_cpos

# python3.7 -m code.cmds.ensemble_dependency evaluate \
# --data "code/data/add_ptb/vi_dev.conllx" \
# -p "./exp3/test_trans_encode" \
# -d 0
