#!/bin/sh

cd ConvE

# train the original model
echo 'Training original model'

CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data FB15k-237 --lr 0.001 --hidden-drop 0.5


echo 'Selecting target triples'
mkdir data/target_conve_FB15k-237_1

CUDA_VISIBLE_DEVICES=0 python -u select_targets.py --model conve --data FB15k-237 --lr 0.001 --hidden-drop 0.5



echo 'Generating random edits for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_1.py --model conve --data FB15k-237 --budget 1 --rand-run 1
python -u wrangle_KG.py rand_add_n_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_add_n_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating global random edits'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_2.py --model conve --data FB15k-237 --budget 1 --rand-run 1
python -u wrangle_KG.py rand_add_g_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_add_g_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5


CUDA_VISIBLE_DEVICES=0 python -u rand_add_attack_2.py --model conve --data FB15k-237 --budget 2 --rand-run 1
python -u wrangle_KG.py rand_add_g_conve_FB15k-237_1_2_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_add_g_conve_FB15k-237_1_2_1 --lr 0.001 --hidden-drop 0.5



echo 'Generating symmetry edits with ground truth minimum'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_1.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py sym_add_1_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data sym_add_1_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating symmetry edits with worse ranks'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_2.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py sym_add_2_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data sym_add_2_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating symmetry edits with cosine distance'
CUDA_VISIBLE_DEVICES=0 python -u sym_add_attack_3.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py sym_add_3_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data sym_add_3_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5





echo 'Generating inversion edits with ground truth minimum'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_1.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py inv_add_1_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data inv_add_1_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating inversion edits with worse ranks'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_2.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py inv_add_2_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data inv_add_2_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating inversion edits with cosine distance'
CUDA_VISIBLE_DEVICES=0 python -u inv_add_attack_3.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py inv_add_3_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data inv_add_3_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5




echo 'Generating composition edits with ground truth values'
python -u create_clusters.py --model conve --data FB15k-237 --num-clusters 300
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_1.py --model conve --data FB15k-237 --budget 1 --num-clusters 300 --rand-run 1
python -u wrangle_KG.py com_add_1_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data com_add_1_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5


echo 'Generating composition attack with just worse ranks '
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_2.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py com_add_2_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data com_add_2_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating composition attack with cosine distance '
CUDA_VISIBLE_DEVICES=0 python -u com_add_attack_3.py --model conve --data FB15k-237 --budget 1
python -u wrangle_KG.py com_add_3_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data com_add_3_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5



echo 'Generating edits from IJCAI-19 baseline '
CUDA_VISIBLE_DEVICES=0 python -u ijcai_add_attack_1.py --model conve --data FB15k-237 --budget 1 --corruption-factor 0.1 --rand-run 1 --use-gpu
python -u wrangle_KG.py ijcai_add_1_conve_FB15k-237_1_1_1_0.1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data ijcai_add_1_conve_FB15k-237_1_1_1_0.1 --lr 0.001 --hidden-drop 0.5


CUDA_VISIBLE_DEVICES=0 python -u ijcai_add_attack_1.py --model conve --data FB15k-237 --budget 1 --corruption-factor 0.3 --rand-run 1 --use-gpu
python -u wrangle_KG.py ijcai_add_1_conve_FB15k-237_1_1_1_0.3
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data ijcai_add_1_conve_FB15k-237_1_1_1_0.3 --lr 0.001 --hidden-drop 0.5



echo 'Generating edits from criage baseline '
python -u wrangle_KG.py target_conve_FB15k-237_1
CUDA_VISIBLE_DEVICES=0 python -u criage_inverter.py --model conve --data FB15k-237 --lr 0.001 --hidden-drop 0.5
CUDA_VISIBLE_DEVICES=0 python -u criage_add_attack_1.py --model conve --data FB15k-237
python -u wrangle_KG.py criage_add_1_conve_FB15k-237_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data criage_add_1_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5












