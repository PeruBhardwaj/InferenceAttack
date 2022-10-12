#!/bin/sh

cd ConvE

echo 'Generating gradient attribution edits with dot similarity : WN18RR DistMult'
python -u grad_add_attack.py --model distmult --data WN18RR --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_distmult_WN18RR_1_1_1
python -u main.py --model distmult --data grad_add_dot_distmult_WN18RR_1_1_1 --lr 0.01 --num-batches 50

echo 'Generating gradient attribution edits with cosine similarity : WN18RR DistMult'
python -u grad_add_attack.py --model distmult --data WN18RR --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_distmult_WN18RR_1_1_1
python -u main.py --model distmult --data grad_add_cos_distmult_WN18RR_1_1_1 --lr 0.01 --num-batches 50

echo 'Generating gradient attribution edits with l2 similarity : WN18RR DistMult'
python -u grad_add_attack.py --model distmult --data WN18RR --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_distmult_WN18RR_1_1_1
python -u main.py --model distmult --data grad_add_l2_distmult_WN18RR_1_1_1 --lr 0.01 --num-batches 50

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : WN18RR ComplEx'
python -u grad_add_attack.py --model complex --data WN18RR --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_complex_WN18RR_1_1_1
python -u main.py --model complex --data grad_add_dot_complex_WN18RR_1_1_1 --lr 0.01

echo 'Generating gradient attribution edits with cosine similarity : WN18RR ComplEx'
python -u grad_add_attack.py --model complex --data WN18RR --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_complex_WN18RR_1_1_1
python -u main.py --model complex --data grad_add_cos_complex_WN18RR_1_1_1 --lr 0.01

echo 'Generating gradient attribution edits with l2 similarity : WN18RR ComplEx'
python -u grad_add_attack.py --model complex --data WN18RR --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_complex_WN18RR_1_1_1
python -u main.py --model complex --data grad_add_l2_complex_WN18RR_1_1_1 --lr 0.01

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : WN18RR ConvE'
python -u grad_add_attack.py --model conve --data WN18RR --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_conve_WN18RR_1_1_1
python -u main.py --model conve --data grad_add_dot_conve_WN18RR_1_1_1 --lr 0.001

echo 'Generating gradient attribution edits with cosine similarity : WN18RR ConvE'
python -u grad_add_attack.py --model conve --data WN18RR --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_conve_WN18RR_1_1_1
python -u main.py --model conve --data grad_add_cos_conve_WN18RR_1_1_1 --lr 0.001

echo 'Generating gradient attribution edits with l2 similarity : WN18RR ConvE'
python -u grad_add_attack.py --model conve --data WN18RR --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_conve_WN18RR_1_1_1
python -u main.py --model conve --data grad_add_l2_conve_WN18RR_1_1_1 --lr 0.001

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : WN18RR TransE'
python -u grad_add_attack.py --model transe --data WN18RR --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_transe_WN18RR_1_1_1
python -u main.py --model transe --data grad_add_dot_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating gradient attribution edits with cosine similarity : WN18RR TransE'
python -u grad_add_attack.py --model transe --data WN18RR --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_transe_WN18RR_1_1_1
python -u main.py --model transe --data grad_add_cos_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12

echo 'Generating gradient attribution edits with l2 similarity : WN18RR TransE'
python -u grad_add_attack.py --model transe --data WN18RR --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_transe_WN18RR_1_1_1
python -u main.py --model transe --data grad_add_l2_transe_WN18RR_1_1_1 --lr 0.005 --input-drop 0.0 --transe-margin 9.0 --num-batches 1000  --epochs 100 --reg-weight 1e-12



