'''
1. Read the processed data (int IDs) and generate sr2o and or2s data from training file
2. Use the train, valid and test file to generate filter lists for evaluation
'''
import numpy as np
import sys
import os
import errno
import json
import pandas as pd
import pickle
from collections import defaultdict


def _load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=None, dtype=str)
    df = df.drop_duplicates()
    return df.values


def generate_eval_filter(dataset_name):
    #processed_path = 'data/processed_{0}'.format(dataset_name)
    processed_path = 'data/{0}'.format(dataset_name)
    files = ['train', 'valid', 'test']
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for file in files:
        file_path = os.path.join(processed_path, file+'.txt')
        examples = _load_data(file_path)
        for lhs, rel, rhs in examples:
            #to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['lhs'][(rhs, rel)].add(int(lhs))  # we don't need reciprocal training
            to_skip['rhs'][(lhs, rel)].add(int(rhs))
    
    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))
            #to_skip_final[kk][(int(k[0]), int(k[1]))] = sorted(list(v))
    
    out = open(os.path.join(processed_path, 'to_skip_eval.pickle'), 'wb')
    pickle.dump(to_skip_final, out)
    out.close()
    
    #with open(os.path.join(processed_path, 'to_skip_eval.json'), 'w') as f:
    #    f.write(json.dumps(to_skip_final)  + '\n')
        
    return

def generate_train_data(dataset_name):
    #processed_path = 'data/processed_{0}'.format(dataset_name)
    processed_path = 'data/{0}'.format(dataset_name)
    file_path = os.path.join(processed_path, 'train.txt')
    train_examples = _load_data(file_path)
    sr2o = defaultdict(set)
    or2s = defaultdict(set)
    for s,r,o in train_examples:
        sr2o[(s,r)].add(o)
        or2s[(o,r)].add(s)
        
    sr2o = {k: sorted(list(v)) for k, v in sr2o.items()}
    or2s = {k: sorted(list(v)) for k, v in or2s.items()}
    
    out = open(os.path.join(processed_path, 'sr2o_train.pickle'), 'wb')
    pickle.dump(sr2o, out)
    out.close()
    
    out = open(os.path.join(processed_path, 'or2s_train.pickle'), 'wb')
    pickle.dump(or2s, out)
    out.close()
    
    #with open(os.path.join(processed_path, 'sr2o_train.json'), 'w') as f:
    #    f.write(json.dumps(sr2o)  + '\n')
    
    #with open(os.path.join(processed_path, 'or2s_train.json'), 'w') as f:
    #    f.write(json.dumps(or2s)  + '\n')
        
    return


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1] # name of dataset
    else:
        #dataset_name = 'FB15k-237'
        #dataset_name = 'YAGO3-10'
        #dataset_name = 'WN18'
        #dataset_name = 'FB15k'
        dataset_name = 'WN18RR'

    seed = 345345
    np.random.seed(seed)
    rdm = np.random.RandomState(seed)
    rng = np.random.default_rng(seed)
    
    print('{}: Generating filter lists for evaluation'.format(dataset_name))
    generate_eval_filter(dataset_name)
    print('{}: Generating train data'.format(dataset_name))
    generate_train_data(dataset_name)
     
        

