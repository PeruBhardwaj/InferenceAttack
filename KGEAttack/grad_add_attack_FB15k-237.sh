#!/bin/sh

cd ConvE

echo 'Generating gradient attribution edits with dot similarity : FB15k-237 DistMult'
python -u grad_add_attack.py --model distmult --data FB15k-237 --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_distmult_FB15k-237_1_1_1
python -u main.py --model distmult --data grad_add_dot_distmult_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

echo 'Generating gradient attribution edits with cosine similarity : FB15k-237 DistMult'
python -u grad_add_attack.py --model distmult --data FB15k-237 --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_distmult_FB15k-237_1_1_1
python -u main.py --model distmult --data grad_add_cos_distmult_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

echo 'Generating gradient attribution edits with l2 similarity : FB15k-237 DistMult'
python -u grad_add_attack.py --model distmult --data FB15k-237 --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_distmult_FB15k-237_1_1_1
python -u main.py --model distmult --data grad_add_l2_distmult_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : FB15k-237 ComplEx'
python -u grad_add_attack.py --model complex --data FB15k-237 --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_complex_FB15k-237_1_1_1
python -u main.py --model complex --data grad_add_dot_complex_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

echo 'Generating gradient attribution edits with cosine similarity : FB15k-237 ComplEx'
python -u grad_add_attack.py --model complex --data FB15k-237 --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_complex_FB15k-237_1_1_1
python -u main.py --model complex --data grad_add_cos_complex_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

echo 'Generating gradient attribution edits with l2 similarity : FB15k-237 ComplEx'
python -u grad_add_attack.py --model complex --data FB15k-237 --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_complex_FB15k-237_1_1_1
python -u main.py --model complex --data grad_add_l2_complex_FB15k-237_1_1_1 --lr 0.005 --input-drop 0.5

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : FB15k-237 ConvE'
python -u grad_add_attack.py --model conve --data FB15k-237 --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_conve_FB15k-237_1_1_1
python -u main.py --model conve --data grad_add_dot_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating gradient attribution edits with cosine similarity : FB15k-237 ConvE'
python -u grad_add_attack.py --model conve --data FB15k-237 --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_conve_FB15k-237_1_1_1
python -u main.py --model conve --data grad_add_cos_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

echo 'Generating gradient attribution edits with l2 similarity : FB15k-237 ConvE'
python -u grad_add_attack.py --model conve --data FB15k-237 --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_conve_FB15k-237_1_1_1
python -u main.py --model conve --data grad_add_l2_conve_FB15k-237_1_1_1 --lr 0.001 --hidden-drop 0.5

# ####################################################################################################################################

echo 'Generating gradient attribution edits with dot similarity : FB15k-237 TransE'
python -u grad_add_attack.py --model transe --data FB15k-237 --reproduce-results --sim-metric dot
python -u wrangle_KG.py grad_add_dot_transe_FB15k-237_1_1_1
python -u main.py --model transe --data grad_add_dot_transe_FB15k-237_1_1_1 --lr 0.001 --input-drop 0.0 --transe-margin 9.0 --num-batches 800 --epochs 100 --reg-weight 1e-10

echo 'Generating gradient attribution edits with cosine similarity : FB15k-237 TransE'
python -u grad_add_attack.py --model transe --data FB15k-237 --reproduce-results --sim-metric cos
python -u wrangle_KG.py grad_add_cos_transe_FB15k-237_1_1_1
python -u main.py --model transe --data grad_add_cos_transe_FB15k-237_1_1_1 --lr 0.001 --input-drop 0.0 --transe-margin 9.0 --num-batches 800 --epochs 100 --reg-weight 1e-10

echo 'Generating gradient attribution edits with l2 similarity : FB15k-237 TransE'
python -u grad_add_attack.py --model transe --data FB15k-237 --reproduce-results --sim-metric l2
python -u wrangle_KG.py grad_add_l2_transe_FB15k-237_1_1_1
python -u main.py --model transe --data grad_add_l2_transe_FB15k-237_1_1_1 --lr 0.001 --input-drop 0.0 --transe-margin 9.0 --num-batches 800 --epochs 100 --reg-weight 1e-10



