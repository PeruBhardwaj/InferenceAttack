#!/usr/bin/env python
# coding: utf-8

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
import pandas as pd
import errno
from sklearn.cluster import MiniBatchKMeans, KMeans

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
    parser.add_argument('--max-norm', action='store_true', help='Option to add unit max norm constraint to entity embeddings')
    
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


parser = add_arguments()
parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')


parser.add_argument('--attack', type=str, default='sym_add_1', help='String to indicate the attack')


# In[4]:




# In[5]:


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


#args.target_split = 1 # which target split to use 
#Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
#args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
#args.rand_run = 1 #  a number assigned to the random run of the experiment

args.seed = args.seed + (args.rand_run - 1) # default seed is 17

#args.model = 'distmult'
#args.data = 'FB15k-237'
#args.attack = 'sym_add_2'
#args.budget = 2

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
elif args.data == 'nations':
    if args.model == 'distmult' or args.model == 'complex':
        args.lr = 0.00001
        args.num_batches = 5
        args.epochs = 500
        args.label_smoothing = 0.0
        args.embedding_dim = 1000
        args.reg_weight = 1e-9
    elif args.model == 'conve':
        args.lr = 0.00001
        args.num_batches = 5
        args.epochs = 500
        args.label_smoothing = 0.0
        args.embedding_dim = 1000
        args.reg_weight = 1e-9
        args.input_drop = 0.3
        args.hidden_drop = 0.5
    elif args.model == 'transe':
        args.lr = 0.00001
        args.num_batches = 5
        args.epochs = 500
        args.label_smoothing = 0.0
        args.embedding_dim = 1000
        args.reg_weight = 1e-5
        args.input_drop = 0.0
        args.transe_margin = 0.1
    else:
        print("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
else:
    print("New dataset:{0}. Set hyperparams".format(args.data))


# In[52]:


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
log_path = save_path = 'logs/decoy_logs/{0}_{1}_{2}_{3}_{4}_{5}.log'.format( args.attack, args.model, 
                                                              args.data, 
                                                              args.target_split, 
                                                              args.budget, 
                                                              args.rand_run
                                                             )
#eval_name = save_path = 'results/decoy/{0}_{1}_{2}_{3}_{4}_{5}.txt'.format( args.attack, args.model, 
#                                                              args.data, 
#                                                              args.target_split, 
#                                                              args.budget, 
#                                                              args.rand_run
#                                                             )


# In[18]:


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename = log_path
                       )
logger = logging.getLogger(__name__)
logger.info(args)
logger.info('-------------------- Eval on decoy triples ----------------------')


data_path = 'data/{0}_{1}_{2}_{3}_{4}_{5}'.format( args.attack, args.model, 
                                                              args.data, 
                                                              args.target_split, 
                                                              args.budget, 
                                                              args.rand_run
                                                             )


# In[19]:


n_ent, n_rel, ent_to_id, rel_to_id = generate_dicts(data_path)


# In[20]:


data  = load_data(data_path)
train_data, valid_data, test_data = data['train'], data['valid'], data['test']


# In[41]:


# Read from decoy file - named decoy_test.txt
df = pd.read_csv(os.path.join(data_path, 'decoy_test.txt'), sep='\t', header=None, names=None, dtype=int)
decoy_data = df.values


# In[42]:


logger.info('Number of triples in test set : {0}'.format(test_data.shape[0]))
logger.info('Number of triples in decoy test : {0}'.format(decoy_data.shape[0]))


# In[43]:


with open(os.path.join(data_path, 'summary_edits.json')) as f:
    summary_dict = json.load(f)


# In[44]:


num_adv_o = 0
num_adv_s = 0
for key, value in summary_dict.items():
    adv_o, adv_s = value[1], value[2]
    if len(adv_o) == 3:
        num_adv_o +=1 
    if len(adv_s) == 3:
        num_adv_s +=1 


# In[38]:


logger.info('Number of adversarial edits on o side: {0}'.format(num_adv_o))
logger.info('Number of adversarial edits on s side: {0}'.format(num_adv_s))


# In[47]:


assert(decoy_data.shape[0] == num_adv_o + num_adv_s)


# In[48]:


decoy_o, decoy_s = decoy_data[:num_adv_o], decoy_data[num_adv_o:]


# In[50]:


logger.info('Number of decoy triples on o side: {0}'.format(decoy_o.shape[0]))
logger.info('Number of decoy triples on s side: {0}'.format(decoy_s.shape[0]))


# In[51]:


# add a model and load the pre-trained params
model = add_model(args, n_ent, n_rel)
model.to(device)
logger.info('Loading saved model from {0}'.format(model_path))
state = torch.load(model_path)
model_params = state['state_dict']
params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
for key, size, count in params:
    logger.info('Key:{0}, Size:{1}, Count:{2}'.format(key, size, count))

model.load_state_dict(model_params)

model.eval()


# In[53]:


# Not using the filter here, because I did not use the filter in inverse edits
def get_ranking(model, queries:torch.Tensor, num_rel:int,
                device: str,
                batch_size: int = 500
               ):
    ranks = []
    ranks_lhs = []
    ranks_rhs = []
    b_begin = 0
    #logger.info('Computing ranks for all queries')
    while b_begin < len(queries):
        b_queries = queries[b_begin : b_begin+batch_size]
        s,r,o = b_queries[:,0], b_queries[:,1], b_queries[:,2]
        r_rev = r+num_rel
        lhs_score = model.forward(o,r_rev, mode='lhs', sigmoid=False) #this gives scores not probabilities
        rhs_score = model.forward(s,r, mode='rhs', sigmoid=False) # this gives scores not probabilities


        # sort and rank
        max_values, lhs_sort = torch.sort(lhs_score, dim=1, descending=True) #high scores get low number ranks
        max_values, rhs_sort = torch.sort(rhs_score, dim=1, descending=True)

        lhs_sort = lhs_sort.cpu().numpy()
        rhs_sort = rhs_sort.cpu().numpy()

        for i, query in enumerate(b_queries):
            # find the rank of the target entities
            lhs_rank = np.where(lhs_sort[i]==query[0].item())[0][0]
            rhs_rank = np.where(rhs_sort[i]==query[2].item())[0][0]

            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks_lhs.append(lhs_rank + 1)
            ranks_rhs.append(rhs_rank + 1)

        b_begin += batch_size

    #logger.info('Ranking done for all queries')
    return np.array(ranks_lhs), np.array(ranks_rhs)


# In[55]:


batch_size = 50


# In[59]:


queries_o = torch.from_numpy(decoy_o.astype('int64')).to(device)
queries_s = torch.from_numpy(decoy_s.astype('int64')).to(device)


# In[57]:


_, ranks_o = get_ranking(model, queries_o, 0, device, batch_size)


# In[60]:


ranks_s, _ = get_ranking(model, queries_s, 0, device, batch_size)


# In[62]:


del model


# In[64]:


pos_data_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format( args.attack, args.model, 
                                                              args.data, 
                                                              args.target_split, 
                                                              args.budget, 
                                                              args.rand_run
                                                             )
if args.attack == 'ijcai_add_1':
    pos_data_name = '{0}_100.0'.format(pos_data_name)
    
pos_model_path = 'saved_models/{0}_{1}.model'.format(pos_data_name, model_name)


# In[65]:


#pos_model_path


# In[66]:


# add a model and load the pre-trained params
pos_model = add_model(args, n_ent, n_rel)
pos_model.to(device)
logger.info('Loading poisoned model from {0}'.format(pos_model_path))
state = torch.load(pos_model_path)
model_params = state['state_dict']
params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
for key, size, count in params:
    logger.info('Key:{0}, Size:{1}, Count:{2}'.format(key, size, count))

pos_model.load_state_dict(model_params)

pos_model.eval()


# In[67]:


_, pos_ranks_o = get_ranking(pos_model, queries_o, 0, device, batch_size)


# In[68]:


pos_ranks_s, _ = get_ranking(pos_model, queries_s, 0, device, batch_size)


# In[84]:


def metrics(ranks):
    hits_at = np.arange(1,11)
    hits_at = list(map(lambda x: np.mean((ranks <= x), dtype=np.float64).item(), 
                                      hits_at))
    
    mr = np.mean(ranks, dtype=np.float64).item()
    
    mrr = np.mean(1. / ranks, dtype=np.float64).item()
    
    return hits_at, mr, mrr


# In[ ]:


# In[89]:


def generate_and_save_results(ranks_orig, ranks_pos, side ='o'):
    #hits, mrr and mr for original ranks - save to logger and file
    orig_hits, orig_mr, orig_mrr = metrics(ranks_orig)
    pos_hits, pos_mr, pos_mrr = metrics(ranks_pos)
    
    hits_at = np.arange(1,11)
    logger.info('------- Metrics for side {0} ------------------------------------'.format(side))
    for i in hits_at:
        logger.info('Original Hits @{0}: {1}'.format(i, orig_hits[i-1]))
        logger.info('Poisoned Hits @{0}: {1}'.format(i, pos_hits[i-1]))
        logger.info('Difference in Hits @{0}: {1}'.format(i,  (orig_hits[i-1] - pos_hits[i-1])))
    logger.info('Original Mean rank : {0}'.format(orig_mr))
    logger.info('Poisoned Mean rank : {0}'.format(pos_mr))
    logger.info('Difference in Mean rank : {0}'.format(orig_mr - pos_mr))
    logger.info('Original Mean reciprocal rank : {0}'.format(orig_mrr))
    logger.info('Poisoned Mean reciprocal rank : {0}'.format(pos_mrr))
    logger.info('Difference in Mean reciprocal rank : {0}'.format(orig_mrr - pos_mrr))
    
    logger.info('-----------------------------------------------------------------')
    
    per_diff_mrr = (orig_mrr - pos_mrr)/ orig_mrr * 100
    
    logger.info('Percent Difference in Mean reciprocal rank : {0}'.format(per_diff_mrr))
    
    logger.info('-----------------------------------------------------------------')
    
    return per_diff_mrr


# In[86]:


per_diff_mrr_o = generate_and_save_results(ranks_o, pos_ranks_o, side='o')


# In[88]:


per_diff_mrr_s = generate_and_save_results(ranks_s, pos_ranks_s, side='o')


# In[87]:

per_diff_mrr = (per_diff_mrr_o + per_diff_mrr_s) / 2
logger.info('Average Percent Difference in Mean reciprocal rank (o+s side): {0}'.format(per_diff_mrr))
logger.info('-----------------------------------------------------------------')

