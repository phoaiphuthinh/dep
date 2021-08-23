#! /bin/bash

python3.7 -m code.cmds.ensemble_dependency train -b -d 0 \
-c "code/config.ini"  \
-p "exp3/test_seperate_train" \
-f bert \
--bert "vinai/phobert-base" \
--train "code/data/ud/vi_train_predicted.conllu" \
--dev "code/data/ud/vi_dev_predicted.conllu" \
--test "code/data/ud/vi_test_predicted.conllu" \
--train_add "code/data/ud/en_gum-ud-train.conllu" \
--dev_add "code/data/ud/en_gum-ud-dev.conllu" \
--test_add "code/data/ud/en_gum-ud-test.conllu" \
--batch-size 1000 \
--use_cpos \
--epochs 400 \
--epochs_add 20 \
--encoder "bert"

# python3.7 -m code.cmds.ensemble_dependency evaluate \
# --data "code/data/add_ptb/vi_dev.conllx" \
# -p "./exp3/test_trans_encode" \
# -d 0
