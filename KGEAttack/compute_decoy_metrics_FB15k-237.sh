#!/bin/sh

cd ConvE

echo 'Computing metrics for decoy in FB15k-237 DistMult '

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data FB15k-237 --attack 'com_add_3' --budget 1 


echo 'Computing metrics for decoy in FB15k-237 Complex'

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data FB15k-237 --attack 'com_add_3' --budget 1 


echo 'Computing metrics for decoy in FB15k-237 Transe '

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data FB15k-237 --attack 'com_add_3' --budget 1 



echo 'Computing metrics for decoy in FB15k-237 ConvE'

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data FB15k-237 --attack 'com_add_3' --budget 1 


















