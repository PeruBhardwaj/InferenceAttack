#!/bin/sh

cd ConvE

# train the original model
echo 'Training original model'


CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data WN18RR --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12


echo 'Selecting target triples'
mkdir data/target_transe_WN18RR_1

CUDA_VISIBLE_DEVICES=0 python -u select_targets.py --model transe --data WN18RR --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Re-training the model to compute baseline change in metrics for target set'
python -u wrangle_KG.py target_transe_WN18RR_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data target_transe_WN18RR_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12


echo 'Generating random edits for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_1.py --model transe --data WN18RR --budget 1 --rand-run 1
python -u wrangle_KG.py rand_add_n_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data rand_add_n_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating global random edits with 1 edit'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_2.py --model transe --data WN18RR --budget 1 --rand-run 1
python -u wrangle_KG.py rand_add_g_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data rand_add_g_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating global random edits with 2 edits'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_2.py --model transe --data WN18RR --budget 2 --rand-run 1
python -u wrangle_KG.py rand_add_g_transe_WN18RR_1_2_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data rand_add_g_transe_WN18RR_1_2_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12



echo 'Generating symmetry edits with ground truth minimum'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_1.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py sym_add_1_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data sym_add_1_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating symmetry edits with worse ranks'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_2.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py sym_add_2_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data sym_add_2_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating symmetry edits with worse ranks'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_3.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py sym_add_3_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data sym_add_3_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12





echo 'Generating inversion edits with ground truth minimum'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_1.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py inv_add_1_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data inv_add_1_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating inversion edits with worse ranks'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_2.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py inv_add_2_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data inv_add_2_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating inversion edits with cosine distance'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_3.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py inv_add_3_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data inv_add_3_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12






echo 'Generating composition edits with ground truth values'
python -u create_clusters.py --model transe --data WN18RR --num-clusters 50
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_1.py --model transe --data WN18RR --budget 1 --num-clusters 50 --rand-run 1
python -u wrangle_KG.py com_add_1_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data com_add_1_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating composition edits with just worse ranks '
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_2.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py com_add_2_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data com_add_2_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating composition edits with cosine distance '
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_3.py --model transe --data WN18RR --budget 1
python -u wrangle_KG.py com_add_3_transe_WN18RR_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data com_add_3_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12




echo 'Generating edits from IJCAI-19 baseline '
CUDA_VISIBLE_DEVICES=0 python -u ijcai_add_attack_1.py --model transe --data WN18RR --budget 1 --corruption-factor 20 --rand-run 1 --use-gpu
python -u wrangle_KG.py ijcai_add_1_transe_WN18RR_1_1_1_20.0
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data ijcai_add_1_transe_WN18RR_1_1_1_20.0 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12


CUDA_VISIBLE_DEVICES=0 python -u ijcai_add_attack_1.py --model transe --data WN18RR --budget 1 --corruption-factor 5 --rand-run 1 --use-gpu
python -u wrangle_KG.py ijcai_add_1_transe_WN18RR_1_1_1_5.0
CUDA_VISIBLE_DEVICES=0 python -u main.py --model transe --data ijcai_add_1_transe_WN18RR_1_1_1_5.0 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12



















