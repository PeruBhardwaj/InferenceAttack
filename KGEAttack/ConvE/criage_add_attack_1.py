#!/usr/bin/env python
# coding: utf-8

# - In this notebook - 
#     -  
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
    

def sig (x, y):
    return 1 / (1 + np.exp(-np.dot(x, np.transpose(y))))

def point_hess(e_o, nei, embd_e, embd_rel):
    H = np.zeros((200, 200))
    for i in nei:
        X = np.multiply(np.reshape(embd_e[i[0]], (1, -1)), np.reshape(embd_rel[i[1]], (1, -1)))
        sig_tri = sig(e_o, X)
        Sig = (sig_tri)*(1-sig_tri)
        H += Sig * np.dot(np.transpose(X), X)
    return H

def point_score(Y, X, e_o, H):
    sig_tri = sig(e_o, X) 
    M = np.linalg.inv(H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X))
    Score = np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, M)))
    return Score, M

def grad_score(Y, X, H, e_o, M):
    grad = []
    n = 200
    sig_tri = sig(e_o, X)
    A = H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X)
    A_in = M 
    X_2 = np.dot(np.transpose(X), X)
    f_part = np.dot(Y, np.dot((1-sig_tri)*np.eye(n)-(sig_tri)*(1-sig_tri)*np.transpose(np.dot(np.transpose(e_o), X)), A_in))
    for i in range(n):
        s = np.zeros((n,n))
        s[:,i] = X
        s[i,:] = X
        s[i,i] = 2*X[0][i]
        Q = np.dot(((sig_tri)*(1-sig_tri)**2-(sig_tri)**2*(1-sig_tri))*e_o[0][i]*X_2+(sig_tri)*(1-sig_tri)*s, A_in)
        grad += [f_part[0][i] - np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, np.dot(A_in, Q))))[0][0]] ######## + 0.02 * X[0][i]]

    return grad

#def find_best_attack(e_o, Y, nei, embd_e, embd_rel, attack_ext, pr):
# parameter attack_ext is not used anywhere inside the function
def find_best_attack(e_o, Y, nei, embd_e, embd_rel, pr):
    H = point_hess(e_o, nei, embd_e, embd_rel)
    X = pr
    step = np.array([[0.00000000001]])
    score = 0 
    score_orig,_ = point_score(Y, pr, e_o,H)
    score_n, M = point_score(Y, X, e_o,H)
    num_iter = 0
    
    atk_flag = 0
    while score_n >= score_orig or num_iter<1:
        if num_iter ==4:
            X = pr
            atk_flag = 1
            print('Returning from find_best_attack without update')
            break
        num_iter += 1
        Grad = grad_score(Y, X, H, e_o, M)
        X = X + step * Grad 
        score = score_n
        score_n, M = point_score(Y, X, e_o, H)

    return X, atk_flag

def find_best_at(pred, E2):
    e2 = E2.view(-1).data.cpu().numpy()
    Pred = pred.view(-1).data.cpu().numpy()
    A1 = np.dot(Pred, e2)
    A2 = np.dot(e2, e2)
    A3 = np.dot(Pred, Pred)
    # I am adding this because I got a math domain error for sqrt (distmult nations)
    a = np.true_divide(A3*A2-0.2, A3*A2-A1**2)
    if a>0 :
        A = math.sqrt(np.true_divide(A3*A2-0.2, A3*A2-A1**2))
    else:
        A = 0
    #A = math.sqrt(np.true_divide(A3*A2-0.2, A3*A2-A1**2))
    B = np.true_divide(math.sqrt(0.2)-A*A1, A2)  
    return float(A), float(B)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data #if isinstance(y, torch.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    #y_one_hot = y_one_hot.view(*(y.shape), -1)
    return torch.autograd.Variable(y_one_hot).cuda() #if isinstance(y, Variable) else y_one_hot




# In[3]:

# In[4]:


parser = add_arguments()
parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')



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
#args.corruption_factor = 0.1


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

# In[7]:


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
log_path = save_path = 'logs/attack_logs/criage_add_1/{0}_{1}_{2}_{3}_{4}.log'.format( args.model, 
                                                              args.data, 
                                                              args.target_split, 
                                                              args.budget, 
                                                              args.rand_run
                                                             )    
    
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename = log_path
                       )
logger = logging.getLogger(__name__)
logger.info(args)
logger.info('-------------------- Edits with CRIAGE baseline ----------------------')

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


# In[8]:

logger.info('Loading pre-trained inverter model')
inverter_model_path = 'saved_models/criage_inverter/{0}_{1}.model'.format(args.data, model_name)
# add a model and load the pre-trained params
model = add_model(args, n_ent, n_rel)
model.to(device)
logger.info('Loading saved model from {0}'.format(inverter_model_path))
state = torch.load(inverter_model_path)
model_params = state['state_dict']
params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
for key, size, count in params:
    logger.info('Key:{0}, Size:{1}, Count:{2}'.format(key, size, count))

model.load_state_dict(model_params)

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.lr_decay)

# One hot encoding buffer that you create out of the loop and just keep reusing
#y_onehot_e1 = torch.FloatTensor(args.batch_size, num_entities)
# One hot encoding buffer that you create out of the loop and just keep reusing
#y_onehot_r = torch.FloatTensor(args.batch_size, num_rel)

model.eval()

#train_data =[]
#with open(os.path.join(data_path, 'train.txt'), 'r') as f:
#    for i, line in enumerate(f):
#        e1, rel, e2 = line.split('\t')
#        train_data += [[e1, rel, e2]]

logger.info('Training data length : {0}'.format((len(train_data)))) 

attack_list = []
E2_list = []
E1_list = []
with open(os.path.join(data_path, 'test.txt'), 'r') as f:
    for i, line in enumerate(f):
        e1, rel, e2 = line.split('\t')
        attack_list += [[int(e1), int(rel), int(e2)]]
        E2_list += [int(e2)]
        E1_list += [int(e1)]


print (len(attack_list))
E2_list = set(E2_list)
E2_dict = {}
E1_list = set(E1_list)
E1_dict = {}

for trip in train_data:
    if trip[2] in E2_list:
        if trip[2] in E2_dict:
            E2_dict[trip[2]] += [[trip[0], trip[1]]]
        else:
            E2_dict[trip[2]] = [[trip[0], trip[1]]]
    if trip[0] in E1_list:
        if trip[0] in E1_dict:
            E1_dict[trip[0]] += [[trip[2], trip[1]]]
        else:
            E1_dict[trip[0]] = [[trip[2], trip[1]]]
            

trip_to_add_o = []
trip_to_add_s = []
embd_e = model.emb_e.weight.data.cpu().numpy()
embd_rel = model.emb_rel.weight.data.cpu().numpy()

n_t = 0
no_atk_found = 0
logger.info("----------------- Generating triples of type s'r'o -----------------")
for trip in attack_list:
    if n_t%500 == 0:
        logger.info('Number of processed triple: '.format(n_t))
        
    n_t += 1
    e1, rel, e2_or = trip[0], trip[1], trip[2]
    e1 = torch.cuda.LongTensor([e1])
    rel = torch.cuda.LongTensor([rel])
    e2 = torch.cuda.LongTensor([e2_or])
    pred = model.encoder(e1, rel)
    E2 = model.encoder_2(e2)
    
    A, B = find_best_at(-pred, E2)
    attack_ext = -A*pred+B*E2
    #if e2_or in E2_dict:
    #    nei = E2_dict[e2_or]
    #    attack, flag = find_best_attack(E2.data.cpu().numpy(), pred.data.cpu().numpy(), nei, embd_e, embd_rel, attack_ext.cpu().detach().numpy())
    #    attack = torch.autograd.Variable(torch.from_numpy(attack)).cuda().float()
        #attack = attack_ext
    #    no_atk_found += flag # flag is 1 when grad update does not happen, 0 otherwise

    #else: 
    #    print('Gradient attack not found for triple number: ', n_t) #this excludes the break inside the function
    #    no_atk_found += 1
    #    attack = attack_ext
    
    attack = attack_ext
    
    E1, R = model.decoder(attack)
    _, predicted_e1 = torch.max(E1, 1)
    _, predicted_R = torch.max(R, 1)
    
    trip_to_add_o.append([predicted_e1.item(), 
                          predicted_R.item(), 
                          e2_or])
    
n_t = 0
logger.info("----------------- Generating triples of type sr'o' ------------------------")
for trip in attack_list:
    if n_t%500 == 0:
        logger.info('Number of processed triple: '.format( n_t))
    
    n_t += 1
    e1_or, rel, e2 = trip[0], trip[1], trip[2]
    e1 = torch.cuda.LongTensor([e1_or])
    rel = torch.cuda.LongTensor([rel])
    e2 = torch.cuda.LongTensor([e2])
    pred = model.encoder(e2, rel)
    E1 = model.encoder_2(e1) 
    
    A, B = find_best_at(-pred, E1)
    attack_ext = -A*pred+B*E1
    #if e1_or in E1_dict:
    #    nei = E1_dict[e1_or]
    #    attack, flag = find_best_attack(E1.data.cpu().numpy(), pred.data.cpu().numpy(), nei, embd_e, embd_rel, attack_ext.cpu().detach().numpy())
    #    attack = torch.autograd.Variable(torch.from_numpy(attack)).cuda().float()
        #attack = attack_ext
    #    no_atk_found += flag # flag is 1 when grad update does not happen, 0 otherwise

    #else: 
    #    print('Gradient attack not found for triple number: ', n_t) #this excludes the break inside the function
    #    no_atk_found += 1
    #    attack = attack_ext
    
    attack = attack_ext
    
    E2, R = model.decoder(attack)
    _, predicted_e2 = torch.max(E2, 1)
    _, predicted_R = torch.max(R, 1)
    
    trip_to_add_s.append([e1_or,
                          predicted_R.item(),
                          predicted_e2.item()
                          ])
    

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


# In[125]:


#decoy_trips = np.concatenate((np.asarray(decoy_trip_o), np.asarray(decoy_trip_s)))


# In[22]:


logger.info ('Length of original test set: ' + str(test_data.shape[0]))
logger.info ('Number of edits generated for o side: ' + str(trips_to_add_o.shape[0]))
logger.info ('Number of edits generated for s side: ' + str(trips_to_add_s.shape[0]))
#logger.info ('Number of triples in decoy test set: ' + str(decoy_trips.shape[0]))


# In[23]:


save_path = 'data/criage_add_1_{0}_{1}_{2}_{3}_{4}'.format( args.model, 
                                                              args.data, 
                                                              args.target_split, 
                                                              args.budget, 
                                                              args.rand_run
                                                             )


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


#with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
#    for item in decoy_trips:
#        out.write("%s\n" % "\t".join(map(str, item)))


# In[31]:


#with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
#    out.write(json.dumps(summary_dict)  + '\n')


# In[35]:


with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
    f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
    f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
    f.write('Length of new poisoned training set including duplicates: {0} \n'. format(new_train_1.shape[0]))
    f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
    f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
    f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
    f.write('Number of triples added from corrupting o_side: {0} (o_dash, r, s)\n'. format(trips_to_add_o.shape[0]))
    f.write('Number of triples added from corrupting s_side: {0} (o, r, s_dash)\n'. format(trips_to_add_s.shape[0]))
    f.write('This attack version is generated using the CRIAGE baseline \n')
    f.write('---------------------------------------------------------------------- \n')
    







