'''
This file contains functions that are used repeatedly across different attacks
'''
import logging
import time
from tqdm import tqdm
import io
import pandas as pd
import numpy as np
import os
import json

import argparse
import torch

from model import Distmult, Complex, Conve, Transe


logger = logging.getLogger(__name__) #config already set in main.py

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

def load_model(model_path, args, n_ent, n_rel, device):
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
    logger.info(model)
    
    return model

def generate_nghbrs(test_set, train_set):
    '''
    For every triple in test set, return the index of 
    neighbouring triple in training set,
    i.e. indices in training set are returned
    '''
    n_dict = {}
    for t, triple in enumerate(test_set):
        sub = triple[0]
        obj = triple[2]
        mask = (np.isin(train_set[:,0], [sub, obj]) | np.isin(train_set[:,2], [sub, obj]))
        #nghbrs_dict[t] = pro_train[mask]
        mask_idx = np.where(mask)[0]
        n_dict[t] = mask_idx
    
    return n_dict 


def perturb_data(train_data, trips_to_delete):
    logger.info('----- Generating perturbed dataset ------')
    per_tr_1 = np.empty_like(train_data)
    per_tr_1[:] = train_data

    n_ignored_edits = 0
    for idx, trip in enumerate(trips_to_delete):
        i = trip[0]
        j = trip[1]
        k = trip[2]
        # mask for triple in training set
        m = (np.isin(per_tr_1[:,0], [i]) & np.isin(per_tr_1[:,1], [j]) & np.isin(per_tr_1[:,2], [k]))
        if np.any(m):
            temp_tr = per_tr_1[~m]
            # mask to check if deleting triple also deletes entity
            m2 = (((np.any(temp_tr[:,0] ==k)) | (np.any(temp_tr[:,2] == k)))
                 & ((np.any(temp_tr[:,0] == i)) | (np.any(temp_tr[:,2] == i))))
            if np.any(m2):
                #np.copyto(per_tr, temp_tr)
                per_tr_1 = np.empty_like(temp_tr)
                per_tr_1[:] = temp_tr
            else:
                n_ignored_edits += 1
                logger.info('Ignoring edit number {0}: {1} because it deletes entities'.format(idx, trip))
        else:
            logger.info('Can\'t delete the selected triple. Something is wrong in the code')
            logger.info(trip)
            break
            
    return per_tr_1, n_ignored_edits


def set_hyperparams(args):
    '''
    Given the args, return with updated hyperparams for reproducibility
    '''
    if args.data == 'WN18RR':
        args.original_data = 'WN18RR'
        
    if args.data == 'WN18':
        args.original_data = 'WN18'
        
    if args.data == 'FB15k-237':
        args.original_data = 'FB15k-237'
        
    if (args.data == 'WN18RR' or args.original_data == 'WN18RR'):
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
            raise Exception("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
    
    if (args.data == 'FB15k-237' or args.original_data == 'FB15k-237'):
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
            raise Exception("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
    
    if (args.data == 'WN18' or args.original_data == 'WN18'):
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
            raise Exception("New model:{0},{1}. Set hyperparams".format(args.data, args.model))
            
            
    return args





def get_argument_parser():
    '''Generate an argument parser
    '''
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
    
    parser.add_argument('--reproduce-results', action='store_true', help='Use the hyperparameters to reproduce the results.')
    parser.add_argument('--original-data', type=str, default='FB15k-237', help='Dataset to use; this option is needed to set the hyperparams to reproduce the results for training after attack, default: FB15k-237')
    
    return parser


class TqdmToLogger(io.StringIO):
    #https://github.com/tqdm/tqdm/issues/313
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)