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

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import TrainDataset, BidirectionalOneShotIterator
from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe



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

def set_paths(args):
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    eval_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.data, model_name, args.num_batches, args.epochs, args.valid_batch_size, args.test_batch_size)
    log_path = 'logs/select_target_{0}_{1}_{2}_{3}_{4}.log'.format(args.data, args.target_split, model_name, args.num_batches, args.epochs)
    
    return model_name, model_path, eval_name, log_path

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
        inp_f = open(os.path.join(data_path, split+'.pickle'), 'rb')
        data[split] = np.array(pickle.load(inp_f))
        inp_f.close()
        
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
    
def get_ranking(model, queries:torch.Tensor, num_rel:int,
               filters:Dict[str, Dict[Tuple[int, int], List[int]]],
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

        for i, query in enumerate(b_queries):
            filter_lhs = filters['lhs'][(query[2].item(), query[1].item())]
            filter_rhs = filters['rhs'][(query[0].item(), query[1].item())]

            # save the prediction that is relevant
            target_value1 = rhs_score[i, query[2].item()].item()
            target_value2 = lhs_score[i, query[0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            lhs_score[i][filter_lhs] = -1e6
            rhs_score[i][filter_rhs] = -1e6
            # write base the saved values
            rhs_score[i][query[2].item()] = target_value1
            lhs_score[i][query[0].item()] = target_value2

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
    return ranks_lhs, ranks_rhs    
    
    
    
    
if __name__ == '__main__':
    parser = add_arguments()
    parser.add_argument('--target-split', type=int, default=1, help='Ranks to use for target set. Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)
    
    args.epochs = -1 #no training here
    model_name, model_path, eval_name, log_path = set_paths(args)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)


    data_path = 'data/{0}'.format(args.data)
    n_ent, n_rel, ent_to_id, rel_to_id = generate_dicts(data_path)
    
    ##### load data####
    data  = load_data(data_path)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']
    
    inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
    to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
    to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}
    
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
    
    with torch.no_grad():
        target_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)
        
        # generate ranks for test set 
        logger.info('Generating target set from test set')
        test_data = torch.from_numpy(test_data.astype('int64')).to(device)
        if args.add_reciprocals:
            num_rel= n_rel
        else:
            num_rel = 0
        ranks_lhs, ranks_rhs = get_ranking(model, test_data, num_rel, to_skip_eval, device, args.test_batch_size)
        ranks_lhs, ranks_rhs = np.array(ranks_lhs), np.array(ranks_rhs)
        #indices_lhs, indices_rhs = np.asarray(ranks_lhs <= 10).nonzero(), np.asarray(ranks_rhs <= 10).nonzero()
        if args.target_split == 2:
            indices = np.asarray(((ranks_lhs <= 100) & (ranks_lhs >10)) & ((ranks_rhs <= 100)&(ranks_rhs > 10))).nonzero()
        elif args.target_split ==1 :
            indices = np.asarray((ranks_lhs <= 10) & (ranks_rhs <= 10)).nonzero()
        else:
            logger.info('Unknown Target Split: {0}', self.args.target_split)
            raise Exception("Unknown target split!")
        
        test_data = test_data.cpu().numpy()
        #targets_lhs, targets_rhs = test_data[indices_lhs], test_data[indices_rhs]
        targets = test_data[indices]
        logger.info('Number of targets generated: {0}'.format(targets.shape[0]))
        #save eval for selected targets
        split = 'target_{0}'.format(args.target_split)
        
        results_target = evaluation(model, targets, to_skip_eval, eval_name, num_rel, split, args.test_batch_size, -1, device)
        # save target set

        with open(os.path.join(target_path, 'target.txt'), 'w') as out:
            for item in targets:
                out.write("%s\n" % "\t".join(map(str, item)))
        with open(os.path.join(target_path, 'test.txt'), 'w') as out:
            for item in targets:
                out.write("%s\n" % "\t".join(map(str, item)))

        # use the valid set to generate non-target set
        logger.info('Generating non target set from valid set')
        valid_data = torch.from_numpy(valid_data.astype('int64')).to(device)
        if args.add_reciprocals:
            num_rel= n_rel
        else:
            num_rel = 0
        ranks_lhs, ranks_rhs = get_ranking(model, valid_data, num_rel, to_skip_eval, device, args.valid_batch_size)
        ranks_lhs, ranks_rhs = np.array(ranks_lhs), np.array(ranks_rhs)
        if args.target_split == 2:
            indices = np.asarray(((ranks_lhs <= 100) & (ranks_lhs >10)) & ((ranks_rhs <= 100)&(ranks_rhs > 10))).nonzero()
        elif args.target_split == 1:
            indices = np.asarray((ranks_lhs <= 10) & (ranks_rhs <= 10)).nonzero()
        else:
            logger.info('Unknown Target Split: {0}', self.args.target_split)
            raise Exception("Unknown target split!")
            
        valid_data = valid_data.cpu().numpy()
        non_targets = valid_data[indices]
        logger.info('Number of non targets generated: {0}'.format(non_targets.shape[0]))
        #save eval for selected non targets
        split = 'non_target_{0}'.format(args.target_split)
        
        results_ntarget = evaluation(model, non_targets, to_skip_eval, eval_name, num_rel, split, args.valid_batch_size, -1, device)
        # save non target set and valid set both - eval needed for both
        with open(os.path.join(target_path, 'non_target.txt'), 'w') as out:
            for item in non_targets:
                out.write("%s\n" % "\t".join(map(str, item)))
        with open(os.path.join(target_path, 'valid.txt'), 'w') as out:
            for item in valid_data:
                out.write("%s\n" % "\t".join(map(str, item)))


        # saving dicts to avoid searching later
        with open(os.path.join(target_path, 'entities_dict.json'), 'w') as f:
            f.write(json.dumps(ent_to_id)  + '\n')

        with open(os.path.join(target_path, 'relations_dict.json'), 'w') as f:
            f.write(json.dumps(rel_to_id)  + '\n')
            
        with open(os.path.join(target_path, 'train.txt'), 'w') as out:
            for item in train_data:
                out.write("%s\n" % "\t".join(map(str, item)))
                
        out = open(os.path.join(target_path, 'to_skip_eval.pickle'), 'wb')
        pickle.dump(to_skip_eval, out)
        out.close()

        # write down the stats for targets generated
        with open(os.path.join(target_path, 'stats.txt'), 'w') as out:
            out.write('Number of train set triples: {0}\n'.format(train_data.shape[0]))
            out.write('Number of test set triples: {0}\n'.format(test_data.shape[0]))
            out.write('Number of valid set triples: {0}\n'.format(valid_data.shape[0]))
            out.write('Number of target triples: {0}\n'.format(targets.shape[0]))
            out.write('Number of non target triples: {0}\n'.format(non_targets.shape[0]))
            if args.target_split ==2:
                out.write('Target triples are ranked >10 and <=100 and test set is the target triples \n')
                out.write('Non target triples are ranked >10 and <=100 but valid triples is original valid set \n')
                out.write('Non target triples with ranks >10 and <=100 are in non_target.txt \n')
            else:
                out.write('Target triples are ranked <=10 and test set is the target triples \n')
                out.write('Non target triples are ranked <=10 but valid triples is original valid set \n')
                out.write('Non target triples with ranks <=10 are in non_target.txt \n')
            out.write('------------------------------------------- \n')





