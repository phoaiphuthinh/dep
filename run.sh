#! /bin/bash

# python3.7 -m code.cmds.ensemble_crf2o train -b -d 0  \
# -c config.ini  \
# -p "exp2/add_ptb_train" \
# -f bert \
# --bert "vinai/phobert-base" \
# --epochs 1000 \
# --train "code/data/add_ptb/vi_train.conllx" \
# --dev "code/data/add_ptb/vi_dev.conllx" \
# --test "code/data/add_ptb/vi_test.conllx" \
# --train_add "code/data/add_ptb/en_full.conllx" \
# --batch-size 5000 \
# --mbr \
# --use_cpos

python3.7 -m code.cmds.ensemble_dependency evaluate \
--data "./code/data/ptb/dev.conllu" \
-p "./exp2/normal_2_thread_nobert" \
-c config.ini \
-d 0
