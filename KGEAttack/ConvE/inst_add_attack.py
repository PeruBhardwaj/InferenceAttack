###  Add triples based on IF triple, chosen as instance similarity - cos, dot, l2
### In inference attacks, 2 additions are selected to decrease the target triple's ranks on s-side and o-side
### But in attribution attack, target triple's rank is reduced (on both s-side and o-side) by selecting the IF triple and adding its corrupted version
### Thus, to integrate Attribution attacks here, I am selecting two IF triples in the neighbourhood and adding their corrputed versions as 2 adversarial additions. Perhaps another version to experiment would be to select the IF triples for s-side and o-side ranks separately and then add their corrupted versions as adversarial additions - the final edits would then be of the form (test_s, test_r', test_o') for o-side, and (test_s', test_r', test_o) for s-side ranks. 

import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import operator

import json
import logging
import argparse 
import math
from pprint import pprint
import errno
import time

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.autograd as autograd

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils

def get_if_triple(test_trip, nghbr_trip, model, attack_batch_size, simmetric):
    test_trip = test_trip[None, :] # add a batch dimension
    test_trip = torch.from_numpy(test_trip).to(device)
    test_s, test_r, test_o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    test_vec = model.score_triples_vec(test_s, test_r, test_o)

    b_begin = 0
    nghbr_sim = []
    if attack_batch_size == -1:
        nghbr_batch = nghbr_trip.shape[0]
    else:
        nghbr_batch = args.attack_batch_size

    while b_begin < nghbr_trip.shape[0]:
        b_nghbr_trip = nghbr_trip[b_begin : b_begin+nghbr_batch]
        b_nghbr_trip = torch.from_numpy(b_nghbr_trip).to(device)
        b_nghbr_s, b_nghbr_r, b_nghbr_o = b_nghbr_trip[:,0], b_nghbr_trip[:,1], b_nghbr_trip[:,2]
        b_nghbr_vec = model.score_triples_vec(b_nghbr_s, b_nghbr_r, b_nghbr_o)
        # shape of nghbr_vec is (num_nghbrs x emb_dim) e.g. (459 x 100)
        # shape of test vec is (1 x emb_dim)
        if simmetric == 'l2':
            b_sim = -torch.norm((b_nghbr_vec-test_vec), p=2, dim=-1)
        elif simmetric == 'dot':
            b_sim = torch.matmul(b_nghbr_vec, test_vec.t()) 
        else: ##cos
            b_sim = F.cosine_similarity(test_vec, b_nghbr_vec) #default dim=1

        b_sim = b_sim.detach().cpu().numpy().tolist()
        nghbr_sim += b_sim
        b_begin += nghbr_batch  

    nghbr_sim = np.array(nghbr_sim)
    nghbr_sim = torch.from_numpy(nghbr_sim).to(device)
    # we want to remove the neighbour with maximum cosine similarity
    max_values, argsort = torch.sort(nghbr_sim, -1, descending=True)
    del_idx_1, del_idx_2 = argsort[0], argsort[1]
    
    return del_idx_1, del_idx_2



def get_additions(train_data, test_data, neighbours, model, attack_batch_size, simmetric):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    if args.model == 'complex':
        ent_emb = torch.cat((model.emb_e_real.weight, model.emb_e_img.weight), dim=-1)
        rel_emb = torch.cat((model.emb_rel_real.weight, model.emb_rel_img.weight), dim=-1)
    else:
        ent_emb = model.emb_e.weight
        rel_emb = model.emb_rel.weight
        
    
    triples_to_delete = []
    triples_to_add = []
    summary_dict = {}
    for test_idx, test_trip in enumerate(test_data):
        test_nghbrs = neighbours[test_idx]
        nghbr_trip = train_data[test_nghbrs]
        del_idx_1, del_idx_2 = get_if_triple(test_trip, nghbr_trip, model, attack_batch_size, simmetric)
        if_trips = [nghbr_trip[del_idx_1], nghbr_trip[del_idx_2]]

        test_trip = torch.from_numpy(test_trip).to(device)[None,:]
        test_s, test_r, test_o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
        
        summary_list = []
        summary_list.append(list(map(int, [test_s.item(),test_r.item(),test_o.item()])))

        for if_trip in if_trips:
            if_trip = torch.from_numpy(if_trip).to(device)[None,:]
            if_s, if_r, if_o = if_trip[:,0], if_trip[:,1], if_trip[:,2]
            
            if (if_o == test_s or if_o == test_o):
                # object of IF triple is neighbour - edit will be [s_dash, if_r, if_o]
                if args.model == 'complex':
                    if_s_emb = torch.cat((model.emb_e_real(if_s), model.emb_e_img(if_s)), dim=-1).squeeze(dim=1)
                else:
                    if_s_emb = model.emb_e(if_s).squeeze(dim=1)
                cos_sim_s = F.cosine_similarity(if_s_emb, ent_emb)
                #cos_sim_r = F.cosine_similarity(if_r_emb, rel_emb)

                # filter for (s_dash, r, o), i.e. ignore s_dash that already exist
                filter_s = train_data[np.where((train_data[:,2] == if_o.item()) 
                                                       & (train_data[:,1] == if_r.item())), 0].squeeze()
                #filter_r = train_data[np.where((train_data[:,0] == if_s.item()) 
                #                                      & (train_data[:,2] == if_o.item())), 1].squeeze()
                cos_sim_s[filter_s] = 1e6
                #cos_sim_r[filter_r] = 1e6

                # sort and rank - smallest cosine similarity means largest cosine distance
                # Hence, corrupted entity = one with smallest cos similarity
                min_values_s, argsort_s = torch.sort(cos_sim_s, -1, descending=False)
                #min_values_r, argsort_r = torch.sort(cos_sim_r, -1, descending=False)
                s_dash = argsort_s[0][None, None]
                #r_dash = argsort_r[0][None, None]

                add_trip = [s_dash.item(), if_r.item(), if_o.item()]

            elif (if_s == test_s or if_s == test_o):
                #print('s is neighbour')
                # subject of IF triple is neighbour - edit will be [if_s, if_r, o_dash]
                if args.model == 'complex':
                    if_o_emb = torch.cat((model.emb_e_real(if_o), model.emb_e_img(if_o)), dim=-1).squeeze(dim=1)
                else:
                    if_o_emb = model.emb_e(if_o).squeeze(dim=1)
                #if_r_emb = model.emb_rel(if_r).squeeze(dim=1)
                cos_sim_o = F.cosine_similarity(if_o_emb, ent_emb)
                #cos_sim_r = F.cosine_similarity(if_r_emb, rel_emb)

                # filter for (s, r, o_dash), i.e. ignore o_dash that already exist
                filter_o = train_data[np.where((train_data[:,0] == if_s.item()) 
                                                       & (train_data[:,1] == if_r.item())), 2].squeeze()
                #filter_r = train_data[np.where((train_data[:,0] == if_s.item()) 
                #                                      & (train_data[:,2] == if_o.item())), 1].squeeze()
                cos_sim_o[filter_o] = 1e6
                #cos_sim_r[filter_r] = 1e6

                # sort and rank - smallest cosine similarity means largest cosine distance
                # Hence, corrupted entity = one with smallest cos similarity
                min_values_o, argsort_o = torch.sort(cos_sim_o, -1, descending=False)
                #min_values_r, argsort_r = torch.sort(cos_sim_r, -1, descending=False)
                o_dash = argsort_o[0][None, None]
                #r_dash = argsort_r[0][None, None]

                add_trip = [if_s.item(), if_r.item(), o_dash.item()]

            else:
                logger.info('Unexpected behaviour')
                
            triples_to_delete.append(if_trip)
            triples_to_add.append(add_trip)
            summary_list.append(list(map(int, add_trip)))
        
        summary_dict[test_idx] = summary_list
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(str(test_idx)))
            logger.info('Time taken: {0}'.format(str(time.time() - start_time)))
    logger.info('Time taken to generate edits: {0}'.format(str(time.time() - start_time)))
    
    return triples_to_delete, triples_to_add, summary_dict

if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--attack-batch-size', type=int, default=-1, help='Batch size for processing neighbours of target')
    
    parser.add_argument('--sim-metric', type=str, default='cos', help='Similarity metric for the attribution attack - cos, dot, l2')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device


    #args.target_split = 1 # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    if args.reproduce_results:
        args = utils.set_hyperparams(args)


    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    args.epochs = -1 #no training here
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    log_path = 'logs/attack_logs/inst_add_{5}/{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run, args.sim_metric)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)


    data_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)

    n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)

    ##### load data####
    data  = utils.load_data(data_path)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']

    inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
    to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
    to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}


    model = utils.load_model(model_path, args, n_ent, n_rel, device)

    neighbours = utils.generate_nghbrs(test_data, train_data) 
    # test set is the target set because we loaded data from target_...

    if_triples, triples_to_add, summary_dict = get_additions(train_data, test_data, neighbours, model, args.attack_batch_size, args.sim_metric)
    
    triples_to_add = np.asarray(triples_to_add)
    if_triples = np.asarray(if_triples)
    
    new_train_1 = np.concatenate((triples_to_add, train_data))
    
    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(new_train_1.shape[0]))
    
    df = pd.DataFrame(new_train_1)
    df = df.drop_duplicates()
    new_train = df.values
    #new_train = new_train_1


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(new_train.shape[0]))
    
    num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
    num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]
    
    save_path = 'data/inst_add_{5}_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run,
                                                               args.sim_metric
                                                              )
    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise
    
    
    with open(os.path.join(save_path, 'train.txt'), 'w') as out:
        for item in new_train:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'train.pickle'), 'wb')
    pickle.dump(new_train.astype('uint64'), out)
    out.close()


    with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
        f.write(json.dumps(ent_to_id)  + '\n')

    with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
        f.write(json.dumps(rel_to_id)  + '\n')
    
    with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
        for item in valid_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
    pickle.dump(valid_data.astype('uint64'), out)
    out.close()

    with open(os.path.join(save_path, 'test.txt'), 'w') as out:
        for item in test_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'test.pickle'), 'wb')
    pickle.dump(test_data.astype('uint64'), out)
    out.close()
    
    with open(os.path.join(save_path, 'influential_triples.txt'), 'w') as out:
        for item in if_triples:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'adversarial_additions.txt'), 'w') as out:
        for item in triples_to_add:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
        out.write(json.dumps(summary_dict)  + '\n')
    
    with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
        f.write('Length of new poisoned training set including duplicates: {0} \n'. format(new_train_1.shape[0]))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('Number of triples addded : {0}\n'.format(triples_to_add.shape[0]))
        #f.write('Number of triples added from corrupting o_side: {0} (o_dash, r, s)\n'. format(trips_to_add_o.shape[0]))
        #f.write('Number of triples added from corrupting s_side: {0} (o, r, s_dash)\n'. format(trips_to_add_s.shape[0]))
        #f.write('In this version, I use reciprocal embedding and its inverse to select (o, r, s_dash)\n')
        f.write('Instance Attribution Attacks - This attack version is generated uses similarity metric: {0} \n'.format(args.sim_metric))
        #f.write('Flag value for maximizing soft truth (If False, minimize): {0}\n' .format(maximize))
        f.write('---------------------------------------------------------------------- \n')



