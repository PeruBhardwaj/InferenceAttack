#!/bin/sh

cd ConvE

echo "Extracting original data.... "
mkdir data/WN18RR_original
mkdir data/FB15k-237_original

tar -xvf WN18RR.tar.gz -C data/WN18RR_original
tar -xvf FB15k-237.tar.gz -C data/FB15k-237_original

echo "Preprocessing... " 
python -u preprocess.py WN18RR
python -u preprocess.py FB15k-237

echo "Wrangling to generate training set and eval filters... "
python -u wrangle_KG.py WN18RR
python -u wrangle_KG.py FB15k-237