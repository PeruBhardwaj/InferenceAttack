#!/usr/bin/env python
# coding: utf-8

# - In this notebook - 
#     - inverse relations are selected using r1.r2 =1 or r1-r2=0 
#     - Select entity substitution that are farthest from target entity o and s in cosine space
#     - (o',ri,s) and (o,ri,s') are the adversarial edits

# In[13]:


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
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F, Parameter

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



# In[ ]:

if __name__ == '__main__':
    parser = add_arguments()
    parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')


    # In[4]:


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # In[5]:


    #args.target_split = 1 # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    #args.model = 'distmult'
    #args.data = 'WN18RR'
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
    else:
        print("New dataset:{0}. Set hyperparams".format(args.data))


    # In[6]:


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
    log_path = 'logs/attack_logs/inv_add_3/{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)

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


    # In[7]:


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






    # In[8]:


    if args.model == 'complex':
        r1 = torch.cat((model.emb_rel_real.weight, model.emb_rel_img.weight), dim=-1)#this is (|R|, 2k)
        r2 = torch.cat((model.emb_rel_real.weight, model.emb_rel_img.weight), dim=-1)#this is (|R|, 2k)

    else:
        r1 = model.emb_rel.weight.data #this is (|R|, k)
        r2 = model.emb_rel.weight.data #this is (|R|, k)


    # In[10]:


    inverse_relation = {}
    if args.model == 'transe':
        # we want r1+r2 ~ 0 for additive models
        inverse_scores = torch.sum(r1[:,None,:] + r2[None, :, :], dim=-1)
    else:
        # we want r1.r2 ~1 for multiplicative models
        inverse_scores = torch.mm(r1, r2.transpose(1,0))
        inverse_scores = 1 - inverse_scores


    inverse_scores = torch.abs(inverse_scores)
    inv_rels = torch.min(inverse_scores, dim=1).indices.cpu().numpy()
    idx = np.arange(r1.shape[0])
    inverse_relation = dict(zip(idx, inv_rels))


    # In[14]:


    trip_to_add_o = []
    trip_to_add_s = []
    decoy_trip_o = []
    decoy_trip_s = []
    summary_dict = {}
    for test_idx, triple in enumerate(test_data):
        triple = torch.from_numpy(triple).to(device)[None,:]
        s,r,o = triple[:,0], triple[:,1], triple[:,2]
        r_inv = torch.from_numpy(np.array([inverse_relation[r.item()]])).to(device)

        # filters are (?,ri, s) and (o,ri,?)
        filter_o = train_data[np.where((train_data[:,2] == s.item()) 
                                               & (train_data[:,1] == r_inv.item())), 0].squeeze()
        filter_s = train_data[np.where((train_data[:,0] == o.item()) 
                                               & (train_data[:,1] == r_inv.item())), 2].squeeze()

        filter_o = torch.from_numpy(filter_o)
        filter_s = torch.from_numpy(filter_s)

        if args.model == 'complex':
            s_embedded = torch.cat((model.emb_e_real(s), model.emb_e_img(s)), dim=-1).squeeze(dim=1)
            o_embedded = torch.cat((model.emb_e_real(o), model.emb_e_img(o)), dim=-1).squeeze(dim=1)
            ent_embedded = torch.cat((model.emb_e_real.weight, model.emb_e_img.weight), dim=-1)
        else:
            s_embedded= model.emb_e(s).squeeze(dim=1)
            o_embedded= model.emb_e(o).squeeze(dim=1)
            ent_embedded = model.emb_e.weight

        # shape of s_embedded and o_embedded is [1,200] i.e. [1,k]
        #print(s_embedded.shape, o_embedded.shape, ent_embedded.shape)
        cos_sim_o = F.cosine_similarity(o_embedded, ent_embedded) # this is 1-d tensor of shape (num_entities)
        cos_sim_s = F.cosine_similarity(s_embedded, ent_embedded)
        #print(cos_sim_o.shape, cos_sim_s.shape)

        # since we want to find entity with minimum cos similarity to target entity, 
        # we set similarity scores for all unwanted entities to large values
        cos_sim_o[filter_o] = 1e6 # this does not contain o from (s,r,o) - but (o,r,s) might exist in filter_o
        cos_sim_o[o] = 1e6 # this is not needed because similarity of o with o will be 1

        cos_sim_s[filter_s] = 1e6
        cos_sim_s[s] = 1e6 # this is not really needed but still 

        # sort and rank - smallest cosine similarity means largest cosine distance
        # Hence, corrupted entity = one with smallest cos similarity
        max_values_o, argsort_o = torch.sort(cos_sim_o, -1, descending=False)
        max_values_s, argsort_s = torch.sort(cos_sim_s, -1, descending=False)

        argsort_o = argsort_o.cpu().numpy()
        argsort_s = argsort_s.cpu().numpy()    

        s,r, r_inv,o = s.item(), r.item(), r_inv.item(), o.item()
        summary_list = []
        summary_list.append(list(map(int, [s,r,o])))

        for bud in range(args.budget):
            o_dash = argsort_o[bud]
            s_dash = argsort_s[bud]
            s_dash, o_dash = s_dash.item(), o_dash.item()

            decoy_o = [s,r,o_dash]
            decoy_s = [s_dash, r, o]

            adv_o = [o_dash, r_inv, s]
            adv_s = [o, r_inv, s_dash]

            # check if adv_o is in training set
            m1 = (np.isin(train_data[:,0], [adv_o[0]]) 
                  & np.isin(train_data[:,1], [adv_o[1]]) 
                  & np.isin(train_data[:,2], [adv_o[2]]))
            if np.any(m1):
                logger.info('Test triple: {0}. Existing triple generated as edit.'.format(test_idx))
                summary_list.append([])
            else:
                decoy_trip_o.append(decoy_o)
                trip_to_add_o.append(adv_o)
                summary_list.append(list(map(int, adv_o)))

            #check if adv_s is in training set
            m2 = (np.isin(train_data[:,0], [adv_s[0]]) 
                  & np.isin(train_data[:,1], [adv_s[1]]) 
                  & np.isin(train_data[:,2], [adv_s[2]]))
            if np.any(m2):
                logger.info('Test triple: {0}. Existing triple generated as edit.'.format(test_idx))
                summary_list.append([])
            else:
                decoy_trip_s.append(decoy_s)
                trip_to_add_s.append(adv_s)
                summary_list.append(list(map(int, adv_s)))

        summary_dict[test_idx] = summary_list
        if test_idx%500 == 0:
            logger.info('Processing test triple : {0}'.format(test_idx))




    # In[15]:


    logger.info(len(trip_to_add_o))
    logger.info(len(trip_to_add_s))
    logger.info(test_data.shape[0])


    # In[16]:


    trips_to_add_o = np.asarray(trip_to_add_o)
    trips_to_add_s = np.asarray(trip_to_add_s)


    # In[17]:


    new_train_1 = np.concatenate((trips_to_add_o, trips_to_add_s, train_data))


    # In[18]:


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(new_train_1.shape[0]))


    # In[19]:


    df = pd.DataFrame(new_train_1)
    df = df.drop_duplicates()
    new_train = df.values
    #new_train = new_train_1


    # In[20]:


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(new_train.shape[0]))


    # In[21]:


    num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
    num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


    # In[22]:


    decoy_trips = np.concatenate((np.asarray(decoy_trip_o), np.asarray(decoy_trip_s)))


    # In[24]:


    logger.info ('Length of original test set: ' + str(test_data.shape[0]))
    logger.info ('Number of edits generated for o side: ' + str(trips_to_add_o.shape[0]))
    logger.info ('Number of edits generated for s side: ' + str(trips_to_add_s.shape[0]))
    logger.info ('Number of triples in decoy test set: ' + str(decoy_trips.shape[0]))


    # In[23]:



    save_path = 'data/inv_add_3_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)


    # In[24]:


    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise


    # In[25]:


    with open(os.path.join(save_path, 'train.txt'), 'w') as out:
        for item in new_train:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'train.pickle'), 'wb')
    pickle.dump(new_train.astype('uint64'), out)
    out.close()


    # In[26]:


    with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
        f.write(json.dumps(ent_to_id)  + '\n')

    with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
        f.write(json.dumps(rel_to_id)  + '\n')


    # In[27]:


    with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
        for item in valid_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
    pickle.dump(valid_data.astype('uint64'), out)
    out.close()


    # In[28]:


    with open(os.path.join(save_path, 'test.txt'), 'w') as out:
        for item in test_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'test.pickle'), 'wb')
    pickle.dump(test_data.astype('uint64'), out)
    out.close()


    # In[29]:


    with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
        for item in decoy_trips:
            out.write("%s\n" % "\t".join(map(str, item)))


    # In[30]:


    with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
        out.write(json.dumps(summary_dict)  + '\n')


    # In[31]:


    inverse_relation = {int(k):int(v) for k,v in inverse_relation.items()}


    # In[32]:


    with open(os.path.join(save_path, 'inverse_relations.json'), 'w') as out:
        out.write(json.dumps(inverse_relation)  + '\n')


    # In[33]:


    with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
        f.write('Length of new poisoned training set including duplicates: {0} \n'. format(new_train_1.shape[0]))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('Number of triples added from corrupting o_side: {0} (o_dash, r, s)\n'. format(trips_to_add_o.shape[0]))
        f.write('Number of triples added from corrupting s_side: {0} (o, r, s_dash)\n'. format(trips_to_add_s.shape[0]))
        f.write('This attack version is generated uses cosine distance between target entity and decoy entity \n')
        f.write('---------------------------------------------------------------------- \n')








