#! /bin/bash

python3.9 -m code.cmds.ensemble_dependency train -b -d 0 \
-c "code/config.ini"  \
-p "test/test" \
-f bert \
--punct \
--tree \
--bert "vinai/phobert-large" \
--train "code/data/ud/vi_train_predicted.conllu" \
--dev "code/data/ud/vi_dev_predicted.conllu" \
--test "code/data/ud/vi_test_predicted.conllu" \
--train_add "code/data/ud/en_gum-train.conllu" \
--dev_add "code/data/ud/en_gum-dev.conllu" \
--test_add "code/data/ud/en_gum-test.conllu" \
--use_cpos \
--batch-size 600 \
--epochs 60 \
--epochs_add 300 \
--encoder "bert"

# python3.7 -m code.cmds.ensemble_dependency evaluate \
# --data "code/data/add_ptb/vi_dev.conllx" \
# -p "./exp3/test_trans_encode" \
# -d 0

# python3.9 -m code.cmds.ensemble_dependency evaluate --punct --data "code/data/org/VnDTv1.1-predicted-POS-tags-test.conll" --buckets 8 --tree -d 0 \
# -p "rate_train_pos_ud_punct_rate5_large/test_seperate_train"