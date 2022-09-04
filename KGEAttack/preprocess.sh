#!/bin/sh

cd ConvE

echo "Adding necessary directories"
mkdir saved_models results losses logs clusters
mkdir saved_models/criage_inverter 
mkdir logs/attack_logs
mkdir logs/attack_logs/criage_add_1 logs/attack_logs/ijcai_add_1 logs/attack_logs/criage_inverter
mkdir logs/attack_logs/rand_add_{n,g} logs/attack_logs/sym_add_{1,2,3} logs/attack_logs/inv_add_{1,2,3} logs/attack_logs/com_add_{1,2,3}


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