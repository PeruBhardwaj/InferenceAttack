import torch
import numpy as np
from torch.autograd import Variable
from sklearn import metrics

import datetime
from typing import Dict, Tuple, List
import logging
import os
import pickle

logger = logging.getLogger(__name__) #config already set in main.py

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
    
    
    
def evaluation(model, queries,  to_skip_eval:Dict[str, Dict[Tuple[int, int], List[int]]], 
               save_name:str, num_rel:int=0, split:str ='test', batch_size:int=500, epoch:int=-1, device:str="cpu"):
    
    
    examples = torch.from_numpy(queries.astype('int64')).to(device)
    
    #get ranking
    ranks_lhs, ranks_rhs = get_ranking(model, examples, num_rel, to_skip_eval, device, batch_size)
    ranks_lhs, ranks_rhs = np.array(ranks_lhs), np.array(ranks_rhs)
    
    #final logging
    hits_at = np.arange(1,11)
    hits_at_lhs = list(map(lambda x: np.mean((ranks_lhs <= x), dtype=np.float64).item(), 
                                      hits_at))
    hits_at_rhs = list(map(lambda x: np.mean((ranks_rhs <= x), dtype=np.float64).item(), 
                                      hits_at))
    mr_lhs = np.mean(ranks_lhs, dtype=np.float64).item()
    mr_rhs = np.mean(ranks_rhs, dtype=np.float64).item()
    
    mrr_lhs = np.mean(1. / ranks_lhs, dtype=np.float64).item()
    mrr_rhs = np.mean(1. / ranks_rhs, dtype=np.float64).item()
    
    
    logger.info('')
    logger.info('-'*50)
    logger.info(split+'_'+save_name)
    logger.info('-'*50)
    logger.info('')
    for i in hits_at:
        logger.info('Hits left @{0}: {1}'.format(i, hits_at_lhs[i-1]))
        logger.info('Hits right @{0}: {1}'.format(i, hits_at_rhs[i-1]))
        logger.info('Hits @{0}: {1}'.format(i, np.mean([hits_at_lhs[i-1],hits_at_rhs[i-1]]).item()))
    logger.info('Mean rank lhs: {0}'.format( mr_lhs))
    logger.info('Mean rank rhs: {0}'.format(mr_rhs))
    logger.info('Mean rank: {0}'.format( np.mean([mr_lhs, mr_rhs])))
    logger.info('Mean reciprocal rank lhs: {0}'.format( mrr_lhs))
    logger.info('Mean reciprocal rank rhs: {0}'.format( mrr_rhs))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean([mrr_rhs, mrr_lhs])))
    
    with open(os.path.join('results', split + '_' + save_name + '.txt'), 'a') as text_file:
        text_file.write('Epoch: {0}\n'.format(epoch))
        text_file.write('Lhs denotes ranking by subject corruptions \n')
        text_file.write('Rhs denotes ranking by object corruptions \n')
        for i in hits_at:
            text_file.write('Hits left @{0}: {1}\n'.format(i, hits_at_lhs[i-1]))
            text_file.write('Hits right @{0}: {1}\n'.format(i, hits_at_rhs[i-1]))
            text_file.write('Hits @{0}: {1}\n'.format(i, np.mean([hits_at_lhs[i-1],hits_at_rhs[i-1]]).item()))
        text_file.write('Mean rank lhs: {0}\n'.format( mr_lhs))
        text_file.write('Mean rank rhs: {0}\n'.format(mr_rhs))
        text_file.write('Mean rank: {0}\n'.format( np.mean([mr_lhs, mr_rhs])))
        text_file.write('MRR lhs: {0}\n'.format( mrr_lhs))
        text_file.write('MRR rhs: {0}\n'.format(mrr_rhs))
        text_file.write('MRR: {0}\n'.format(np.mean([mrr_rhs, mrr_lhs])))
        text_file.write('-------------------------------------------------\n')
        
        
    results = {}
    for i in hits_at:
        results['hits_lhs@{}'.format(i)] = hits_at_lhs[i-1]
        results['hits_rhs@{}'.format(i)] = hits_at_rhs[i-1]
    results['mrr_lhs'] = mrr_lhs
    results['mrr_rhs'] = mrr_rhs
    results['mr_lhs'] = mr_lhs
    results['mr_rhs'] = mr_rhs
    
    return results
        
    
    
   
        