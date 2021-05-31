
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
from criage_model import Distmult, Conve


# In[2]:


def add_arguments():
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    
    parser.add_argument('--data', type=str, default='FB15k-237', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex, transe}')
    parser.add_argument('--add-reciprocals', action='store_true', help='Option to add reciprocal relations')
    
    
    parser.add_argument('--transe-margin', type=float, default=12.0, help='Margin value for TransE scoring function. Default:12.0')
    parser.add_argument('--transe-norm', type=int, default=2, help='P-norm value for TransE scoring function. Default:2')
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')#maybe 0.1
    parser.add_argument('--lr-decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--max-norm', action='store_true', help='Option to add unit max norm constraint to entity embeddings')
    
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches for training (default: 400)') #maybe 200?
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
    
    
    parser.add_argument('--reg-weight', type=float, default=0.0, help='Weight for regularization. Default: 5e-12')#maybe 5e-2?
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
    
def load_train_data(data_path, args, n_rel):
    ##### train    #####
    inp_f = open(os.path.join(data_path, 'sr2o_train.pickle'), 'rb')
    sr2o_train: Dict[Tuple[int, int], List[int]] = pickle.load(inp_f)
    inp_f.close()

    inp_f = open(os.path.join(data_path, 'or2s_train.pickle'), 'rb')
    or2s_train: Dict[Tuple[int, int], List[int]] = pickle.load(inp_f)
    inp_f.close()

    if args.add_reciprocals:
        # adding reciprocals
        or2s_train = {(int(k[0]), int(k[1])+n_rel): v for k,v in or2s_train.items()}
    else:
        or2s_train = {(int(k[0]), int(k[1])): v for k,v in or2s_train.items()}
    sr2o_train = {(int(k[0]), int(k[1])): v for k,v in sr2o_train.items()}
    
    return sr2o_train, or2s_train
    
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


# In[5]:


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


# In[6]:


#args.target_split = 1 # which target split to use 
#Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
#args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
#args.rand_run = 1 #  a number assigned to the random run of the experiment

args.seed = args.seed + (args.rand_run - 1) # default seed is 17

#args.model = 'distmult'
#args.data = 'FB15k-237'


# In[7]:


# Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)


#args.epochs = -1 #no training here
model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
#log_path = 'logs/inv_add_1_{0}_{1}_{2}_{3}.log'.format(args.data, model_name, args.num_batches, args.epochs)
log_path = save_path = 'logs/attack_logs/criage_inverter/{0}_{1}_{2}_{3}'.format( args.data,
                                                                                         model_name, 
                                                                                         args.num_batches, 
                                                                                         args.epochs
                                                                                         )    
    
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename = log_path
                       )
logger = logging.getLogger(__name__)
logger.info(args)
logger.info('-------------------- Running Criage Inverter ----------------------')


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


logger.info('Loading training data')
sr2o_train, or2s_train = load_train_data(data_path, args, n_rel)

# lhs denotes subject side corruptions and rhs denotes object side corruptions
batch_size_lhs = math.ceil(len(list(or2s_train.keys()))/args.num_batches)
batch_size_rhs = math.ceil(len(list(sr2o_train.keys()))/args.num_batches)

logger.info("Dict size or2s:{0}".format(len(list(or2s_train.keys()))))
logger.info('Batch_size_lhs: {0}'.format(batch_size_lhs))
logger.info("Dict size sr2o:{0}".format(len(list(sr2o_train.keys()))))
logger.info('Batch_size_rhs: {0}'.format(batch_size_rhs))

train_dataloader_lhs = DataLoader(
            TrainDataset(args, n_ent, or2s_train, mode='lhs'),
            batch_size      = batch_size_lhs,
            shuffle         = True,
            num_workers     = 0, #max(0, args.num_workers),
            collate_fn      = TrainDataset.collate_fn
            )

train_dataloader_rhs = DataLoader(
            TrainDataset(args, n_ent, sr2o_train, mode='rhs'),
            batch_size      = batch_size_rhs,
            shuffle         = True,
            num_workers     = 0, #max(0, self.args.num_workers),
            collate_fn      = TrainDataset.collate_fn
            )


# In[8]:

logger.info('Loading pre-trained model params')
# add a model and load the pre-trained params
model = add_model(args, n_ent, n_rel)
model.to(device)
logger.info('Loading saved model from {0}'.format(model_path))
model_state = model.state_dict()
pre_state = torch.load(model_path)
pretrained = pre_state['state_dict']
for name in model_state:
    if name in pretrained:
        model_state[name].copy_(pretrained[name])


#model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)

logger.info('----- Training -----')
for epoch in range(args.epochs):
    model.train()
    train_iterator = BidirectionalOneShotIterator(train_dataloader_lhs, train_dataloader_rhs)
    losses = []
    for b in range(2*args.num_batches):
        optimizer.zero_grad()
        batch = next(train_iterator)
        e1, rel,label,mode = batch
        e1, rel = e1.to(device), rel.to(device)
        E1, R = model.forward(e1, rel)
        loss_E1 = model.loss(E1, e1) #e1.squeeze(1))
        loss_R = model.loss(R, rel) #rel.squeeze(1))
        loss = loss_E1 + loss_R
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (b%100 == 0) or (b== (2*args.num_batches-1)):
            logger.info('[E:{} | {}]: Train Loss:{:.4}'.format(epoch, b, np.mean(losses)))
            
    loss = np.mean(losses)
    logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
    

logger.info('Saving trained inverter model')
save_path = 'saved_models/criage_inverter/{0}_{1}.model'.format(args.data, model_name)
state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }
torch.save(state, save_path)
logger.info('Saving model to {0}'.format(save_path))
        
        
    




