#!/bin/sh

cd ConvE

echo 'Computing metrics for decoy in WN18RR DistMult '

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model distmult --data WN18RR --attack 'com_add_3' --budget 1 


echo 'Computing metrics for decoy in WN18RR Complex'

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model complex --data WN18RR --attack 'com_add_3' --budget 1 


echo 'Computing metrics for decoy in WN18RR Transe '

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model transe --data WN18RR --attack 'com_add_3' --budget 1 



echo 'Computing metrics for decoy in WN18RR ConvE'

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'sym_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'sym_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'sym_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'inv_add_1' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'inv_add_2' --budget 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'inv_add_3' --budget 1

CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'com_add_1' --budget 1 --rand-run 1
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'com_add_2' --budget 1 
CUDA_VISIBLE_DEVICES=0 python -u decoy_test.py --model conve --data WN18RR --attack 'com_add_3' --budget 1 

















