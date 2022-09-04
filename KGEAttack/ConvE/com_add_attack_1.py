#!/usr/bin/env python
# coding: utf-8

# - In this notebook - 
#     - composition of a relation is selected by taking the Euclidean distance between r and r1.r2 (for multiplicative models) and r1+r2 (for additive models)
#     - select decoy triple that has minimum soft truth grounding
#     - Select entity substitution based on the t-norm of (s,r1,o'') ^ (o'',r2,o')
#     - use the maximum t-norm for edit 
#     - (s,r1,o'') and (o'',r2,o') are the adversarial edits
#     
# 

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
import time
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

# In[4]:


parser = add_arguments()
parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
parser.add_argument('--num-clusters', type=int, default=100, help='Number of clusters to be generated')
parser.add_argument('--num-cluster-run', type=int, default=1, help='Which cluster run to use?')


# In[5]:


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


#args.target_split = 1 # which target split to use 
#Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
#args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
#args.rand_run = 1 #  a number assigned to the random run of the experiment

args.seed = args.seed + (args.rand_run - 1) # default seed is 17

#args.model = 'distmult'
#args.data = 'FB15k-237'
#args.num_clusters = 200


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


# In[8]:


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
log_path = 'logs/attack_logs/com_add_1/{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)
    
    
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


# In[9]:


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


# In[ ]:
logger.info('------ Generating edits per target triple ------')
start_time = time.time()
logger.info('Start time: {0}'.format(str(start_time)))





# In[10]:


if args.model == 'complex':
    r1 = torch.cat((model.emb_rel_real.weight.data, model.emb_rel_img.weight.data), dim=-1)#this is (|R|, 2k)
    r2 = torch.cat((model.emb_rel_real.weight.data, model.emb_rel_img.weight.data), dim=-1)#this is (|R|, 2k)
else:
    r1 = model.emb_rel.weight.data #this is (|R|, k)
    r2 = model.emb_rel.weight.data #this is (|R|, k)


# In[11]:


r1,r2 = r1.cpu().numpy(), r2.cpu().numpy()


# In[12]:


# all possible compositions of relations
composed_rel = []
for relation in r1:
    if args.model == 'transe':
        comp = relation + r2
    else:
        comp = relation * r2
    composed_rel.append(comp)
composed_rel = np.array(composed_rel) 
composed_rel = composed_rel.reshape(-1,composed_rel.shape[-1]) # shape is (|R|x|R|, k)
# the above reshape will have all rows for r10 composed with full r2, then r11 with r2 and so on
# (composition operations are commutative; so not much worry needed for reshape ordering)


# In[13]:


logger.info(composed_rel.shape)


# In[14]:


composition_relation = {}
for rel_id, relation in enumerate(r1):
    dist = relation - composed_rel
    euclidean_dist = np.linalg.norm(dist, axis=1) # shape is (|R|x|R|)
    euclidean_dist = euclidean_dist.reshape(n_rel, -1) # shape is (|R| , |R|)
    
    candidate_r = np.unravel_index(np.argmin(euclidean_dist), shape=euclidean_dist.shape)
    composition_relation[rel_id] = candidate_r
    
    if rel_id % 100 ==0:
        logger.info('Processing relation number: {0} '.format(rel_id))


# In[ ]:




# In[15]:

if args.data == 'countries_S1' or args.data == 'countries_S2' or args.data == 'countries_S3' or args.data == 'nations':
    cluster_en = np.arange(n_ent)

else:
    cluster_path = 'clusters/{0}_{1}_{2}_{3}'.format( args.model, args.data, args.num_clusters, args.num_cluster_run)


    # In[17]:


    inp_f = open(cluster_path + 'labels.pickle', 'rb')
    labels = np.array(pickle.load(inp_f))
    inp_f.close()


    # In[33]:


    # select a random entity from each cluster
    cluster_en = []
    for i in range(args.num_clusters):
        candidates = np.where(labels == i)[0]
        en = rng.choice(a=candidates, size = 1, replace=True)[0]
        cluster_en.append(en)
    cluster_en = np.array(cluster_en)


# In[46]:



# **Choosing the decoy triple using the minimum soft truth grounding**
# 
# Grounding for composition is $\mathcal{G}_t: (s, \mathtt{r_1}, o'') \wedge (o'', \mathtt{r_2}, o') \Rightarrow (s, \mathtt{r}, o')$.
# 
# The soft truth score is $\phi(\mathcal{G}_t) = \phi(s,r_1,o'') \cdot \phi(o'',r2,o') \cdot \phi(s,r,o') - \phi(s,r_1,o'') \cdot \phi(o'',r_2,o') + 1$
# 
# $\phi(\mathcal{G}_t) = \phi(s',r_1,s'') \cdot \phi(s'',r2,o) \cdot \phi(s',r,o) - \phi(s',r_1,s'') \cdot \phi(s'',r_2,o) + 1$
# 
# 
# For a chosen decoy triple $(s, \mathtt{r}, o')$, I can find the entity by choosing $\phi(\mathcal{G}_t) = \phi(s,r_1,o'') \cdot \phi(o'',r2,o')$ with maximum value
# 

# In[102]:


trip_to_add_o = []
trip_to_add_s = []
decoy_trip_o = []
decoy_trip_s = []
summary_dict = {}
for test_idx, triple in enumerate(test_data):
    triple = torch.from_numpy(triple).to(device)[None,:]
    s,r,o = triple[:,0], triple[:,1], triple[:,2]

    r1 = torch.from_numpy(np.array([composition_relation[r.item()][0]])).to(device)
    r2 = torch.from_numpy(np.array([composition_relation[r.item()][1]])).to(device)
    
    # filters are (?,r, o) and (s,r,?)
    # filters are needed to filter out decoy triples
    filter_o = to_skip_eval['rhs'][(s.item(), r.item())]
    filter_s = to_skip_eval['lhs'][(o.item(), r.item())]
    
    o_dash_vals = torch.zeros(cluster_en.shape[0]*args.budget)
    o_dash_idx = torch.zeros(cluster_en.shape[0]*args.budget)
    for i, en in enumerate(cluster_en):
        en = torch.tensor([en]).to(device)
        # for selecting o''
        # compute scores for (s,r1,o'')
        #score_r1 = model.score_triples(s,r1,en, sigmoid=True).squeeze()
        # compute scores for (o'',r2,o')
        score_r2 = model.forward(en,r2,mode='rhs',sigmoid=True).squeeze()
        # compute scores for (s,r,o')
        score_r = model.forward(s,r,mode='rhs',sigmoid=True).squeeze()
        #target_truth = score_r - 1
        #ground_truth = (score_r1 * score_r2 * target_truth) + 1
        ground_truth = (score_r2 * score_r) - score_r2
        ground_truth[filter_o] = 1e6
        ground_truth[o] = 1e6
        o_dash = torch.topk(ground_truth, k=args.budget, largest=False)
        o_dash_vals[i] = o_dash.values
        o_dash_idx[i] = o_dash.indices
        #o_dash_idx.append(o_dash.indices[:args.budget])
    
    idx = torch.topk(o_dash_vals, k=args.budget, largest=False).indices
    o_dash_a = o_dash_idx[idx].long()
    
    s_dash_vals = torch.zeros(cluster_en.shape[0]*args.budget)
    s_dash_idx = torch.zeros(cluster_en.shape[0]*args.budget)
    for i, en in enumerate(cluster_en):
        en = torch.tensor([en]).to(device)
        # for selecting o''
        # compute scores for (s',r1,o'')
        score_r1 = model.forward(en,r1,mode='lhs',sigmoid=True).squeeze()
        # compute scores for (o'',r2,o)
        #score_r2 = model.score_triples(en,r2,o, sigmoid=True).squeeze()
        # compute scores for (s',r,o)
        score_r = model.forward(o,r,mode='lhs',sigmoid=True).squeeze()
        #target_truth = score_r - 1
        #ground_truth = (score_r1 * score_r2 * target_truth) + 1
        ground_truth = (score_r1 * score_r) - score_r1
        ground_truth[filter_s] = 1e6
        ground_truth[s] = 1e6
        s_dash = torch.topk(ground_truth, k=args.budget, largest=False)
        s_dash_vals[i] = s_dash.values
        s_dash_idx[i] = s_dash.indices
        #o_dash_idx.append(o_dash.indices[:args.budget])
    
    idx = torch.topk(s_dash_vals, k=args.budget, largest=False).indices
    s_dash_a = s_dash_idx[idx].long()
    
    o_dash_a, s_dash_a = o_dash_a.to(device), s_dash_a.to(device)
    
    summary_list = []
    summary_list.append(list(map(int, [s.item(),r.item(),o.item()])))
        
    for bud in range(args.budget):
        o_dash = o_dash_a[bud][None]
        # filters are (s,r1,?) and (?,r2, o') 
        # these filters are needed because we don't want to add back existing edits
        filter_r = train_data[np.where((train_data[:,0] == s.item()) 
                                           & (train_data[:,1] == r1.item())), 2].squeeze()
        filter_l = train_data[np.where((train_data[:,2] == o_dash.item()) 
                                           & (train_data[:,1] == r2.item())), 0].squeeze()
        
        pred_r = model.forward(s,r1, mode='rhs', sigmoid=True) #this gives probabilities
        pred_l = model.forward(o_dash,r2, mode='lhs', sigmoid=True) # this gives probabilities
        pred_r, pred_l = pred_r.squeeze(), pred_l.squeeze()
        
        soft_scores = pred_l * pred_r
        soft_scores[filter_l] = -1e6
        soft_scores[filter_r] = -1e6
        #soft_scores[s] = -1e6
        soft_scores[o] = -1e6 # we don't want to select (s,r1,o) as edit
        
        o_ddash = torch.argmax(soft_scores)
        
        s_dash = s_dash_a[bud][None]
        # filters are (s',r1,?) and (?,r2, o) 
        filter_r = train_data[np.where((train_data[:,0] == s_dash.item()) 
                                           & (train_data[:,1] == r1.item())), 2].squeeze()
        filter_l = train_data[np.where((train_data[:,2] == o.item()) 
                                           & (train_data[:,1] == r2.item())), 0].squeeze()
        
        pred_r = model.forward(s_dash,r1, mode='rhs', sigmoid=True) #this gives probabilities
        pred_l = model.forward(o,r2, mode='lhs', sigmoid=True) # this gives probabilities
        pred_r, pred_l = pred_r.squeeze(), pred_l.squeeze()
        
        soft_scores = pred_l * pred_r
        soft_scores[filter_l] = -1e6
        soft_scores[filter_r] = -1e6
        soft_scores[s] = -1e6 # we don't want to select (s,r2,o) as edit
        #soft_scores[o] = -1e6 
        
        s_ddash = torch.argmax(soft_scores)
        _s,_r, _r1, _r2, _o = s.item(), r.item(), r1.item(), r2.item(), o.item()
        
        s_dash, o_dash = s_dash.item(), o_dash.item()
        o_ddash, s_ddash = o_ddash.item(), s_ddash.item()

        decoy_o = [_s, _r, o_dash]
        decoy_s = [s_dash, _r, _o]
        
        adv_o_l = [_s, _r1,  o_ddash]
        adv_o_r = [o_ddash, _r2, o_dash]
        adv_s_l = [s_dash, _r1,  s_ddash]
        adv_s_r = [s_ddash, _r2,  _o]
        
        decoy_trip_o.append(decoy_o)
        trip_to_add_o.append(adv_o_l)
        trip_to_add_o.append(adv_o_r)
        summary_list.append(list(map(int, adv_o_l)))
        summary_list.append(list(map(int, adv_o_r)))
        
        decoy_trip_s.append(decoy_s)
        trip_to_add_s.append(adv_s_l)
        trip_to_add_s.append(adv_s_r)
        summary_list.append(list(map(int, adv_s_l)))
        summary_list.append(list(map(int, adv_s_r)))
        
    
    summary_dict[test_idx] = summary_list
    if test_idx%500 == 0:
        logger.info('Processing test triple : {0}'.format(test_idx))


# In[103]:
logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))

logger.info(len(trip_to_add_o))
logger.info(len(trip_to_add_s))
logger.info(test_data.shape[0])


# In[104]:


trips_to_add_o = np.asarray(trip_to_add_o)
trips_to_add_s = np.asarray(trip_to_add_s)


# In[105]:


new_train_1 = np.concatenate((trips_to_add_o, trips_to_add_s, train_data))


# In[106]:


logger.info ('Length of original training set: ' + str(train_data.shape[0]))
logger.info ('Length of new poisoned training set: ' + str(new_train_1.shape[0]))


# In[107]:


df = pd.DataFrame(new_train_1)
df = df.drop_duplicates()
new_train = df.values
#new_train = new_train_1


# In[108]:


logger.info ('Length of original training set: ' + str(train_data.shape[0]))
logger.info ('Length of new poisoned training set: ' + str(new_train.shape[0]))


# In[109]:


num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


# In[125]:


decoy_trips = np.concatenate((np.asarray(decoy_trip_o), np.asarray(decoy_trip_s)))


# In[126]:


logger.info ('Length of original test set: ' + str(test_data.shape[0]))
logger.info ('Number of edits generated for o side: ' + str(trips_to_add_o.shape[0]))
logger.info ('Number of edits generated for s side: ' + str(trips_to_add_s.shape[0]))
logger.info ('Number of triples in decoy test set: ' + str(decoy_trips.shape[0]))


# In[113]:


save_path = 'data/com_add_1_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)


# In[114]:


try :
    os.makedirs(save_path)
except OSError as e:
    if e.errno == errno.EEXIST:
        logger.info(e)
        logger.info('Using the existing folder {0} for processed data'.format(save_path))
    else:
        raise


# In[115]:


with open(os.path.join(save_path, 'train.txt'), 'w') as out:
    for item in new_train:
        out.write("%s\n" % "\t".join(map(str, item)))

out = open(os.path.join(save_path, 'train.pickle'), 'wb')
pickle.dump(new_train.astype('uint64'), out)
out.close()


# In[116]:


with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
    f.write(json.dumps(ent_to_id)  + '\n')

with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
    f.write(json.dumps(rel_to_id)  + '\n')


# In[117]:


with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
    for item in valid_data:
        out.write("%s\n" % "\t".join(map(str, item)))

out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
pickle.dump(valid_data.astype('uint64'), out)
out.close()


# In[118]:


with open(os.path.join(save_path, 'test.txt'), 'w') as out:
    for item in test_data:
        out.write("%s\n" % "\t".join(map(str, item)))
        
out = open(os.path.join(save_path, 'test.pickle'), 'wb')
pickle.dump(test_data.astype('uint64'), out)
out.close()


# In[127]:


with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
    for item in decoy_trips:
        out.write("%s\n" % "\t".join(map(str, item)))


# In[120]:


with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
    out.write(json.dumps(summary_dict)  + '\n')


# In[121]:


composition_relation = {int(k):(int(v[0]), int(v[1])) for k,v in composition_relation.items()}


# In[122]:


with open(os.path.join(save_path, 'composed_relations.json'), 'w') as out:
    out.write(json.dumps(composition_relation)  + '\n')


# In[124]:


with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
    f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
    f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
    f.write('Length of new poisoned training set including duplicates: {0} \n'. format(new_train_1.shape[0]))
    f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
    f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
    f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
    f.write('Number of triples added from corrupting o_side: {0} (o_dash, r, s)\n'. format(trips_to_add_o.shape[0]))
    f.write('Number of triples added from corrupting s_side: {0} (o, r, s_dash)\n'. format(trips_to_add_s.shape[0]))
    f.write('This attack version is generated for decoy chosen via minimum ground truth \n')
    f.write('---------------------------------------------------------------------- \n')
    
