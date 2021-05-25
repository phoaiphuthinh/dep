#! /bin/bash

python3.7 -m train --feat bert --build -d 0 \
--train "" \
--dev "" \
--test "" \
--buckets 32 \
--punct \
--train_add "" \
--dev_add "" \
--test_add "" \
--n_embed 100 \
--embed "./glove.6B.txt" \
--bert "vinai/phobert" \