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

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import TrainDataset, BidirectionalOneShotIterator
from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe

class Main(object):
    def __init__(self, args):
        self.args = args
        
        self.model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
        #leaving batches from the model_name since they do not depend on model_architecture 
        # also leaving kernel size and filters, siinice don't intend to change those
        self.model_path = 'saved_models/{0}_{1}.model'.format(args.data, self.model_name)
        self.log_path = 'logs/{0}_{1}_{2}_{3}.log'.format(args.data, self.model_name, args.num_batches, args.epochs)
        self.eval_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.data, self.model_name, args.num_batches, args.epochs, args.valid_batch_size, args.test_batch_size)
        self.loss_path = 'losses/{0}_{1}_{2}_{3}.pickle'.format(args.data, self.model_name, args.num_batches, args.epochs)
        
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = self.log_path
                           )
        self.logger = logging.getLogger(__name__)
        self.logger.info(vars(self.args))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.num_workers = self.args.num_workers if torch.cuda.is_available() else 0
        
        #if self.args.model == 'transe':
        #    self.args.label_smoothing = 0.0
        self.load_data()
        self.model        = self.add_model()
        self.optimizer    = self.add_optimizer(self.model.parameters())
        
        
    def _load_data(self, file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=None, dtype=str)
        df = df.drop_duplicates()
        return df.values
    
    def load_data(self):
        '''
        Load the eval filters and train data
        Make the TrainDataset
        Make head and tail iterators
        '''
        data_path = 'data/{0}'.format(self.args.data)
        #inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
        #self.to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        #inp_f.close()
        
        with open (os.path.join(data_path, 'entities_dict.json'), 'r') as f:
            self.ent_to_id = json.load(f)
        with open (os.path.join(data_path, 'relations_dict.json'), 'r') as f:
            self.rel_to_id = json.load(f)
        self.n_ent = len(list(self.ent_to_id.keys()))
        self.n_rel = len(list(self.rel_to_id.keys()))
        
        ##### train    #####
        inp_f = open(os.path.join(data_path, 'sr2o_train.pickle'), 'rb')
        self.sr2o_train: Dict[Tuple[int, int], List[int]] = pickle.load(inp_f)
        inp_f.close()
        
        inp_f = open(os.path.join(data_path, 'or2s_train.pickle'), 'rb')
        self.or2s_train: Dict[Tuple[int, int], List[int]] = pickle.load(inp_f)
        inp_f.close()
        
        
        if self.args.add_reciprocals:
            # adding reciprocals
            self.or2s_train = {(int(k[0]), int(k[1])+self.n_rel): v for k,v in self.or2s_train.items()}
        else:
            self.or2s_train = {(int(k[0]), int(k[1])): v for k,v in self.or2s_train.items()}
        self.sr2o_train = {(int(k[0]), int(k[1])): v for k,v in self.sr2o_train.items()}
        
        # lhs denotes subject side corruptions and rhs denotes object side corruptions
        batch_size_lhs = math.ceil(len(list(self.or2s_train.keys()))/self.args.num_batches)
        batch_size_rhs = math.ceil(len(list(self.sr2o_train.keys()))/self.args.num_batches)
        
        self.logger.info("Dict size or2s:{0}".format(len(list(self.or2s_train.keys()))))
        self.logger.info('Batch_size_lhs: {0}'.format(batch_size_lhs))
        self.logger.info("Dict size sr2o:{0}".format(len(list(self.sr2o_train.keys()))))
        self.logger.info('Batch_size_rhs: {0}'.format(batch_size_rhs))
        
        self.train_dataloader_lhs = DataLoader(
                    TrainDataset(self.args, self.n_ent, self.or2s_train, mode='lhs'),
                    batch_size      = batch_size_lhs,
                    shuffle         = True,
                    num_workers     = max(0, self.args.num_workers),
                    collate_fn      = TrainDataset.collate_fn
                    )
        
        self.train_dataloader_rhs = DataLoader(
                    TrainDataset(self.args, self.n_ent, self.sr2o_train, mode='rhs'),
                    batch_size      = batch_size_rhs,
                    shuffle         = True,
                    num_workers     = max(0, self.args.num_workers),
                    collate_fn      = TrainDataset.collate_fn
                    )
        
        #self.train_iterator = BidirectionalOneShotIterator(train_dataloader_lhs, train_dataloader_rhs)
        
        ##### test and valid ####
        self.valid_data = self._load_data(os.path.join(data_path, 'valid.txt'))
        self.test_data = self._load_data(os.path.join(data_path, 'test.txt'))
    
        inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
        self.to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        self.to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in self.to_skip_eval['lhs'].items()}
        self.to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in self.to_skip_eval['rhs'].items()}
        
        
    def add_model(self):
        if self.args.add_reciprocals:
            if self.args.model is None:
                model = Conve(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'conve':
                model = Conve(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'distmult':
                model = Distmult(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'complex':
                model = Complex(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'transe':
                model = Transe(self.args, self.n_ent, 2*self.n_rel)
            else:
                self.logger.info('Unknown model: {0}', self.args.model)
                raise Exception("Unknown model!")
        else:
            if self.args.model is None:
                model = Conve(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'conve':
                model = Conve(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'distmult':
                model = Distmult(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'complex':
                model = Complex(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'transe':
                model = Transe(self.args, self.n_ent, self.n_rel)
            else:
                self.logger.info('Unknown model: {0}', self.args.model)
                raise Exception("Unknown model!")
        
        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        #if self.args.optimizer == 'adam' : return torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.lr_decay)
        #else                    : return torch.optim.SGD(parameters,  lr=self.args.lr, weight_decay=self.args.lr_decay)
        return torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.lr_decay)
        
        
    def save_model(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.args)
        }
        torch.save(state, self.model_path)
        self.logger.info('Saving model to {0}'.format(self.model_path))
        
        
    def load_model(self):
        self.logger.info('Loading saved model from {0}'.format(self.model_path))
        state = torch.load(self.model_path)
        model_params = state['state_dict']
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            self.logger.info(key, size, count)
        
        self.model.load_state_dict(model_params)
        self.optimizer.load_state_dict(state['optimizer'])
        
        
    def evaluate(self, split, batch_size, epoch):
        # run the evaluation - 'split.txt' will be loaded and used for ranking
        self.model.eval()
        with torch.no_grad():
            if self.args.add_reciprocals:
                num_rel= self.n_rel
            else:
                num_rel = 0
            
            if split == 'test':
                results = evaluation(self.model, self.test_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            elif split == 'valid':
                results = evaluation(self.model, self.valid_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            else:
                data_path = 'data/{0}'.format(self.args.data)
                inp_f = open(os.path.join(data_path, split+'.pickle'), 'rb')
                split_data = np.array(pickle.load(inp_f))
                inp_f.close()
                results = evaluation(self.model, split_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            
            
            self.logger.info('[Epoch {} {}]: MRR: lhs : {:.5}, rhs : {:.5}, Avg : {:.5}'.format(epoch, split, results['mrr_lhs'], results['mrr_rhs'], np.mean([results['mrr_lhs'], results['mrr_rhs']])))
        # evaluation has its own logging; so no need to log here
        return results
    
    
    def regularizer(self):
        # Apply p-norm regularization; assign weights to each param
        weight = self.args.reg_weight
        p = self.args.reg_norm
        
        if self.args.model == 'complex':
            trainable_params = [self.model.emb_e_real.weight, self.model.emb_e_img.weight, self.model.emb_rel_real.weight, self.model.emb_rel_img.weight]
        else:
            trainable_params = [self.model.emb_e.weight, self.model.emb_rel.weight]
        
        norm = 0
        for i in range(len(trainable_params)):
            #norm += weight * trainable_params[i].norm(p = p)**p
            norm += weight * torch.sum( torch.abs(trainable_params[i]) ** p)
            
        return norm
        #return norm / trainable_params[0].shape[0] #KBC codebase
                
        
    
    def run_epoch(self, epoch):
        train_iterator = BidirectionalOneShotIterator(self.train_dataloader_lhs, self.train_dataloader_rhs)
        self.model.train()
        losses = []
        self.logger.info("length of dataloader lhs: {0}".format(len(iter(self.train_dataloader_lhs))))
        self.logger.info("length of dataloader rhs: {0}".format(len(iter(self.train_dataloader_rhs))))
        for b in range(2*self.args.num_batches):
            self.optimizer.zero_grad()
            batch = next(train_iterator)
            s,r,label,mode = batch
            s,r = s.to(self.device), r.to(self.device)
            
            pred = self.model.forward(s,r,mode=mode)
            #self.logger.info("pred shape: ({0}, {1})".format(pred.shape[0], pred.shape[1]))
            label = label.to(self.device)
            loss = self.model.loss(pred, label)
            if self.args.reg_weight != 0.0:
                loss  += self.regularizer()
            
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            if (b%100 == 0) or (b== (2*self.args.num_batches-1)):
                self.logger.info('[E:{} | {}]: Train Loss:{:.4}'.format(epoch, b, np.mean(losses)))
                
        
        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss
    
    def fit(self):
        if self.args.resume:
            self.load_model()
            results = self.evaluate(split=self.args.resume_split, batch_size = self.args.test_batch_size, epoch = -1)
            pprint(results)
        else:
            self.model.init() #this initializes embeddings using Xavier
            
        self.logger.info(self.model)
        
        train_losses = []
        for epoch in range(self.args.epochs):
            
            train_loss = self.run_epoch(epoch)
            train_losses.append(train_loss)
            self.save_model()
            
            if epoch%20 == 0:
                results_valid = self.evaluate(split='valid', batch_size = self.args.valid_batch_size, epoch = epoch)
                results_test = self.evaluate(split='test', batch_size = self.args.test_batch_size, epoch = epoch)
            if epoch == (self.args.epochs - 1):
                results_valid = self.evaluate(split='valid', batch_size = self.args.valid_batch_size, epoch = epoch)
                results_test = self.evaluate(split='test', batch_size = self.args.test_batch_size, epoch = epoch)
                # save train losses
                with open(self.loss_path, "wb") as fp:   #Pickling
                    pickle.dump(train_losses, fp)
                #with open("test.txt", "rb") as fp:   # Unpickling
                #    b = pickle.load(fp)
                
            
        
        return
    
    
if __name__ == '__main__':
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
    

    args = parser.parse_args()
    
    np.set_printoptions(precision=3)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = Main(args)
    model.fit()


            
            
        

