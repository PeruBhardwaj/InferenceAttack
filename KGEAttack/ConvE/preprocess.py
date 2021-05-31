'''
1. Read the string data and generate dictionaries
2. Convert string data to processed data and save it along with dictionaries
'''

import numpy as np
import sys
import os
import errno
import json
import pandas as pd
import pickle


if len(sys.argv) > 1:
    dataset_name = sys.argv[1] # name of dataset
else:
    #dataset_name = 'FB15k-237'
    #dataset_name = 'YAGO3-10'
    #dataset_name = 'WN18'
    #dataset_name = 'FB15k'
    dataset_name = 'WN18RR'

seed = 345345
np.random.seed(seed)
rdm = np.random.RandomState(seed)
rng = np.random.default_rng(seed)

base_path = 'data/{0}_original/'.format(dataset_name)
#processed_path = 'data/processed_{0}'.format(dataset_name)
processed_path = 'data/{0}'.format(dataset_name)
files = ['train', 'valid', 'test']

def _load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=None, dtype=str)
    df = df.drop_duplicates()
    return df.values

def generate_ids():
    complete_data = []
    for file in files:
        file_path = os.path.join(base_path, file+'.txt')
        complete_data.append(_load_data(file_path))
            
    complete_data = np.concatenate(complete_data)
    unique_ent = np.unique(np.concatenate((complete_data[:,0], complete_data[:,2])))
    unique_rel = np.unique(complete_data[:,1])
    
    entities_to_id = {x:i for (i,x) in enumerate(sorted(unique_ent))}
    rel_to_id = {x:i for (i,x) in enumerate(sorted(unique_rel))}
    
    print("{} entities and {} relations".format(len(unique_ent), len(unique_rel)))
    
    return unique_ent, unique_rel, entities_to_id, rel_to_id 

def generate_ids_from_train():
    file_path = os.path.join(base_path, 'train.txt')
    X_train = _load_data(file_path)
    #complete_dataset = np.concatenate(complete_dataset)
    unique_ent = np.unique(np.concatenate((X_train[:, 0], X_train[:, 2])))
    unique_rel = np.unique(X_train[:, 1])
    
    entities_to_id = {x: i for (i, x) in enumerate(sorted(unique_ent))}
    rel_to_id = {x: i for (i, x) in enumerate(sorted(unique_rel))}
    
    print("{} entities and {} relations".format(len(unique_ent), len(unique_rel)))
    
    return unique_ent, unique_rel, entities_to_id, rel_to_id


def process_and_save(entities_to_id, relations_to_id, unique_ent):
    try :
        os.makedirs(processed_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print('Using the existing folder {0} for processed data'.format(processed_path))
        else:
            raise
            
    with open(os.path.join(processed_path, 'dataset_stats.txt'), 'w') as file:
        file.write("{} entities and {} relations \n".format(len(unique_ent), len(unique_rel)))
    
    # function to filter out triples with unseen entities
    def _filter_unseen_entities(x):
        ent_seen = unique_ent
        df = pd.DataFrame(x, columns=['s', 'p', 'o'])
        filtered_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
        n_removed_ents = df.shape[0] - filtered_df.shape[0]
        return filtered_df.values, n_removed_ents
        
    
    for f in files:
        file_path = os.path.join(base_path, f+'.txt')
        x = _load_data(file_path)
        x, n_removed_ents = _filter_unseen_entities(x) # filter unseen entities if any
        if n_removed_ents > 0:
            msg = '{0} split: Removed {1} triples containing unseen entities. \n'.format(f, n_removed_ents)
            with open(os.path.join(processed_path, 'dataset_stats.txt'), 'a') as file:
                file.write(msg)
            print(msg)
        x_idx_s = np.vectorize(entities_to_id.get)(x[:, 0])
        x_idx_p = np.vectorize(relations_to_id.get)(x[:, 1])
        x_idx_o = np.vectorize(entities_to_id.get)(x[:, 2]) 
          
        x = np.dstack([x_idx_s, x_idx_p, x_idx_o]).reshape((-1, 3))
        
        with open(os.path.join(processed_path, f+'.txt'), 'w') as out:
            for item in x:
                out.write("%s\n" % "\t".join(map(str, item)))
          
        out = open(os.path.join(processed_path, f+'.pickle'), 'wb')
        pickle.dump(x.astype('uint64'), out)
        out.close()
        
    return


filter_unseen = True # if needed, pass as cmd argument
if filter_unseen:
    unique_ent, unique_rel, entities_to_id, rel_to_id = generate_ids_from_train()
else:
    unique_ent, unique_rel, entities_to_id, rel_to_id = generate_ids()
      
n_relations = len(unique_rel)
n_entities = len(unique_ent)

    
process_and_save(entities_to_id, rel_to_id, unique_ent)

    
with open(os.path.join(processed_path, 'entities_dict.json'), 'w') as f:
    f.write(json.dumps(entities_to_id)  + '\n')
    
with open(os.path.join(processed_path, 'relations_dict.json'), 'w') as f:
    f.write(json.dumps(rel_to_id)  + '\n')
    
print("{} entities and {} relations".format(len(unique_ent), len(unique_rel)))
    
    

    