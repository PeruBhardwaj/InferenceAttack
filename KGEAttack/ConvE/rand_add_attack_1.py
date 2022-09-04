#!/usr/bin/env python
# coding: utf-8

# - In this notebook - 
#     - generate random edits of type (s,r',o') and (s',r',o) 
#     - this is equivalent to the Rand_5 in ConvE_7.0

# In[1]:


import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import json
import torch
import logging
import argparse 
import math
from pprint import pprint
import errno
import time

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import TrainDataset, BidirectionalOneShotIterator
from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe


# In[2]:


def add_arguments():
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    
    parser.add_argument('--data', type=str, default='FB15k-237', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--add-reciprocals', action='store_true', help='Option to add reciprocal relations')
    
    
    parser.add_argument('--transe-margin', type=float, default=12.0, help='Margin value for TransE scoring function. Default:12.0')
    parser.add_argument('--transe-norm', type=int, default=2, help='P-norm value for TransE scoring function. Default:2')
    
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train (default: 400)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')#maybe 0.1
    parser.add_argument('--lr-decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    
    parser.add_argument('--num-batches', type=int, default=400, help='Number of batches for training (default: 400)') #maybe 200?
    parser.add_argument('--test-batch-size', type=int, default=128, help='Batch size for test split (default: 128)')
    parser.add_argument('--valid-batch-size', type=int, default=128, help='Batch size for valid split (default: 128)')    
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to use for the batch loaders on GPU. Default: 4')
    
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    
    parser.add_argument('--stack_width', type=int, default=20, help='The first dimension of the reshaped/stacked 2D embedding. Second dimension is inferred. Default: 20')
    #parser.add_argument('--stack_height', type=int, default=10, help='The second dimension of the reshaped/stacked 2D embedding. Default: 10')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('-num-filters', default=32,   type=int, help='Number of filters for convolution')
    parser.add_argument('-kernel-size', default=3, type=int, help='Kernel Size for convolution')
    
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    
    
    parser.add_argument('--reg-weight', type=float, default=5e-12, help='Weight for regularization. Default: 5e-12')#maybe 5e-2?
    parser.add_argument('--reg-norm', type=int, default=2, help='Norm for regularization. Default: 3')
    
    parser.add_argument('--resume', action='store_true', help='Restore a saved model.')
    parser.add_argument('--resume-split', type=str, default='test', help='Split to evaluate a restored model')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='Random seed (default: 17)')
    
    return parser


def generate_dicts(data_path):
    with open (os.path.join(data_path, 'entities_dict.json'), 'r') as f:
        ent_to_id = json.load(f)
    with open (os.path.join(data_path, 'relations_dict.json'), 'r') as f:
        rel_to_id = json.load(f)
    n_ent = len(list(ent_to_id.keys()))
    n_rel = len(list(rel_to_id.keys()))
    
    return n_ent, n_rel, ent_to_id, rel_to_id

import pandas as pd
def load_data(data_path):
    data = {}
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(os.path.join(data_path, split+'.txt'), sep='\t', header=None, names=None, dtype=int)
        df = df.drop_duplicates()
        data[split] = df.values
        
    return data
    
def add_model(args, n_ent, n_rel):
    if args.add_reciprocals:
        if args.model is None:
            model = Conve(args, n_ent, 2*n_rel)
        elif args.model == 'conve':
            model = Conve(args, n_ent, 2*n_rel)
        elif args.model == 'distmult':
            model = Distmult(args, n_ent, 2*n_rel)
        elif args.model == 'complex':
            model = Complex(args, n_ent, 2*n_rel)
        elif args.model == 'transe':
            model = Transe(args, n_ent, 2*n_rel)
        else:
            logger.info('Unknown model: {0}', args.model)
            raise Exception("Unknown model!")
    else:
        if args.model is None:
            model = Conve(args, n_ent, n_rel)
        elif args.model == 'conve':
            model = Conve(args, n_ent, n_rel)
        elif args.model == 'distmult':
            model = Distmult(args, n_ent, n_rel)
        elif args.model == 'complex':
            model = Complex(args, n_ent, n_rel)
        elif args.model == 'transe':
            model = Transe(args, n_ent, n_rel)
        else:
            logger.info('Unknown model: {0}', args.model)
            raise Exception("Unknown model!")

    #model.to(self.device)
    return model
    


# In[3]:



# In[4]:

if __name__ == '__main__':
    parser = add_arguments()
    parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget per target per corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')


    # In[5]:


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # In[8]:


    #args.target_split = 1 # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    #args.model = 'distmult'
    #args.data = 'FB15k-237'
    # Below is based on hyperparams for original model
    if args.data == 'WN18RR':
        if args.model == 'distmult':
            args.lr = 0.01
            args.num_batches = 50
        elif args.model == 'complex':
            args.lr = 0.01
        elif args.model == 'conve':
            args.lr =  0.001
        elif args.model == 'transe':
            args.lr = 0.005 
            args.input_drop = 0.0 
            args.transe_margin = 9.0
            args.num_batches = 1000  
            args.epochs = 100
            args.reg_weight = 1e-12
        else:
            print("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
    elif args.data == 'FB15k-237':
        if args.model == 'distmult':
            args.lr = 0.005
            args.input_drop = 0.5
        elif args.model == 'complex':
            args.lr = 0.005
            args.input_drop = 0.5
        elif args.model == 'conve':
            args.lr = 0.001 
            args.hidden_drop = 0.5
        elif args.model == 'transe':
            args.lr = 0.001 
            args.input_drop = 0.0 
            args.transe_margin = 9.0 
            args.num_batches = 800 
            args.epochs = 100
            args.reg_weight = 1e-10
        else:
            print("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
    elif args.data == 'WN18':
        if args.model == 'distmult':
            args.lr = 0.01
            args.num_batches = 50
        elif args.model == 'complex':
            args.lr = 0.01
        elif args.model == 'conve':
            args.lr =  0.005
        elif args.model == 'transe':
            args.lr = 0.01 
            args.input_drop = 0.0 
            args.transe_margin = 9.0
            args.num_batches = 1500  
            args.epochs = 100
            args.reg_weight = 1e-12
        else:
            print("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
    else:
        print("New dataset:{0}. Set hyperparams".format(args.data))


    # In[9]:


    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    args.epochs = -1 #no training here
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    #log_path = 'logs/inv_add_1_{0}_{1}_{2}_{3}.log'.format(args.data, model_name, args.num_batches, args.epochs)
    log_path = 'logs/attack_logs/rand_add_n/{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)


    data_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)

    n_ent, n_rel, ent_to_id, rel_to_id = generate_dicts(data_path)

    ##### load data####
    data  = load_data(data_path)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']

    inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
    to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
    to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}


    # In[ ]:





    # #### Pseudocode 
    #  - Sample an entity from list of entities and relation from list of relations (the list should exclude target entity and relation)
    #  - If the random triple already exists in the data, sample agains

    # In[10]:


    ents = np.asarray(list(ent_to_id.values()))
    rels = np.asarray(list(rel_to_id.values()))


    # In[11]:


    logger.info(train_data.shape)


    # In[12]:
    
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))


    # Finding corruptions of type 1; checking existence in train set and adding them
    per_tr = np.empty_like(train_data)
    per_tr[:] = train_data

    summary_dict_o = {}
    logger.info('------------ Object Side corruptions -----------------')
    trip_to_add_o = [] # of type sr`o`
    test_trip_idx = 0
    while test_trip_idx < len(test_data):
        if test_trip_idx%500 == 0:
            logger.info('Processing triple ' + str(test_trip_idx))

        test_trip = test_data[test_trip_idx]
        s = test_trip[0]
        r = test_trip[1]
        o = test_trip[2]
        #print(s,r,o)

        budget_idx = 0 
        # depending on how many triples already exist in data and how many triples are added, this might turn into an infinite loop
        while budget_idx < args.budget:

            rel_choices = rels[np.where(rels!=r)]
            # o needs to be excluded because the target entities should not be linked by KGE eval protocols
            ent_choices = ents[np.where(ents!=o)]

            rand_r1 = rng.choice(a=rel_choices, size = 1, replace=True)[0]
            rand_o = rng.choice(a=ent_choices, size = 1, replace=True)[0]

            adv_o = [s, rand_r1, rand_o]
            # mask for  triple in training set
            m1 = (np.isin(per_tr[:,0], [adv_o[0]]) 
                  & np.isin(per_tr[:,1], [adv_o[1]]) 
                  & np.isin(per_tr[:,2], [adv_o[2]]))
            if np.any(m1):
                print('Random triple already exists...generating another random triple')
            else:
                trip_to_add_o.append(adv_o)
                per_tr = np.append(per_tr, np.asarray(adv_o).reshape(-1,3), axis=0)
                summary_dict_o[(s,r,o)] = tuple(adv_o)
                budget_idx += 1

        test_trip_idx += 1


    summary_dict_s = {}
    logger.info('------------ Subject Side corruptions -----------------')
    trip_to_add_s = [] # of type sr`o`
    test_trip_idx = 0
    while test_trip_idx < len(test_data):
        if test_trip_idx%500 == 0:
            logger.info('Processing triple ' + str(test_trip_idx))
        test_trip = test_data[test_trip_idx]
        s = test_trip[0]
        r = test_trip[1]
        o = test_trip[2]
        #print(s,r,o)

        budget_idx = 0
        while budget_idx < args.budget:
            rel_choices = rels[np.where(rels!=r)]
            ent_choices = ents[np.where(ents!=o)]

            rand_r2 = rng.choice(a=rel_choices, size = 1, replace=True)[0]
            rand_s = rng.choice(a=ent_choices, size = 1, replace=True)[0]

            adv_s = [rand_s, rand_r2, o]
            # mask for  triple in training set
            m2 = (np.isin(per_tr[:,0], [adv_s[0]]) 
                  & np.isin(per_tr[:,1], [adv_s[1]]) 
                  & np.isin(per_tr[:,2], [adv_s[2]]))
            if np.any(m2):
                logger.info('Random triple already exists...generating another random triple')
            else:
                trip_to_add_s.append(adv_s)
                per_tr = np.append(per_tr, np.asarray(adv_s).reshape(-1,3), axis=0)
                summary_dict_s[(s,r,o)] = tuple(adv_s)
                budget_idx += 1 

        test_trip_idx += 1

    del per_tr
    
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))


    # In[13]:


    summary_dict = {'rhs':summary_dict_o, 'lhs':summary_dict_s}


    # In[15]:


    logger.info(len(trip_to_add_o))
    logger.info(len(trip_to_add_s))
    logger.info(test_data.shape[0])


    # In[13]:


    trips_to_add_o = np.asarray(trip_to_add_o)
    trips_to_add_s = np.asarray(trip_to_add_s)


    # In[14]:


    new_train = np.concatenate((trips_to_add_o, trips_to_add_s, train_data))


    # In[15]:


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(new_train.shape[0]))


    # In[16]:


    num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
    num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


    # In[17]:


    #decoy_trips = np.concatenate((np.asarray(decoy_trip_o), np.asarray(decoy_trip_s)))


    # In[18]:


    logger.info ('Length of original test set: ' + str(test_data.shape[0]))
    logger.info ('Number of edits generated for o side: ' + str(trips_to_add_o.shape[0]))
    logger.info ('Number of edits generated for s side: ' + str(trips_to_add_s.shape[0]))
    #print ('Number of triples in decoy test set: ' + str(decoy_trips.shape[0]))


    # In[19]:


    save_path = 'data/rand_add_n_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)


    # In[20]:


    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise


    # In[21]:


    with open(os.path.join(save_path, 'train.txt'), 'w') as out:
        for item in new_train:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'train.pickle'), 'wb')
    pickle.dump(new_train.astype('uint64'), out)
    out.close()


    # In[22]:


    with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
        f.write(json.dumps(ent_to_id)  + '\n')

    with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
        f.write(json.dumps(rel_to_id)  + '\n')


    # In[23]:


    with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
        for item in valid_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
    pickle.dump(valid_data.astype('uint64'), out)
    out.close()


    # In[24]:


    with open(os.path.join(save_path, 'test.txt'), 'w') as out:
        for item in test_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'test.pickle'), 'wb')
    pickle.dump(test_data.astype('uint64'), out)
    out.close()


    # In[25]:


    #with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
    #    for item in decoy_trips:
    #        out.write("%s\n" % "\t".join(map(str, item)))


    # In[26]:


    #cannot use this because keyys cannot be tuples for json
    #with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
    #    out.write(json.dumps(summary_dict)  + '\n')


    # In[27]:


    out = open(os.path.join(save_path, 'summary_edits.pickle'), 'wb')
    pickle.dump(summary_dict, out)
    out.close()


    # In[28]:


    with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('Number of triples added from corrupting o_side: {0} (o_dash, r, s)\n'. format(trips_to_add_o.shape[0]))
        f.write('Number of triples added from corrupting s_side: {0} (o, r, s_dash)\n'. format(trips_to_add_s.shape[0]))
        f.write('This attack version is generated from neighbourhood random edits \n')
        f.write('---------------------------------------------------------------------- \n')


