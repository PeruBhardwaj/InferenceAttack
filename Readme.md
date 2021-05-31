## Poisoning Knowledge Graph Embeddings via Relation Inference Patterns
This is the code repository to accompany ACL-2021 paper 'Poisoning Knowledge Graph Embeddings via Relation Inference Patterns'.

Below we describe the structure of the code repository, dependencies and steps to run the experiments.


### Code structure
- Commandline instructions for all experiments are available in bash scripts at this level
- The main codebase is in ConvE
    - script to preprocess data (generate dictionaries) is preprocess.py
    - script to generate evaluation filters and training tuples is wrangle_KG.py
    - script to train a KGE model is main.py
    - script to select target triples from the test set is select_targets.py
    - Random neighbourhood baseline is in rand_add_attack_1.py
    - Random global baseline is in rand_add_attack_2.py
    - Zhang et al. baseline is implemented in ijcai_add_attack.py
    - CRIAGE baseline is in criage_add_attack_1.py
    - Proposed symmetric attacks in sym_add_attack
        - 1 for soft truth score 
        - 2 for KGE ranks 
        - 3 for cosine distance
    - Proposed inversion attacks in inv_add_attack
        - 1 for soft truth score 
        - 2 for KGE ranks 
        - 3 for cosine distance
    - Proposed composition attacks in com_add_attack
        - 1 for soft truth score 
        - 2 for KGE ranks 
        - 3 for cosine distance
    - Elbow method to select clusters is in clustering_elbow.ipynb
    - Script to generate clusters is create_clusters.py
    - Script to compute metrics on decoy set in decoy_test.py
    - Folder elbow_plots contains the elbow plots
    - Folder data will contain datasets generated from running the experiments. 
        - These are named as attack_model_dataset_split_budget_run 
        - here split=1 for target split, budget=1 for most attacks except random global with 2 edits, and run is the number for a random run
        - For Zhang et al. attacks, an additional argument is down sampling percent
    - Folder saved_models, clusters, logs, results and losses are also empty but will be used if a script is run
    


### Dependencies
- python = 3.8.5
- pytorch = 1.4.0
- numpy = 1.19.1
- jupyter = 1.0.0
- pandas = 1.1.0
- matplotlib = 3.2.2
- scikit-learn = 0.23.2
- seaborn = 0.11.0

We have also included the conda environment file inference_attack.yml


### Reproducing the results
- To preprocess the original dataset, use the bash script preprocess.sh
- Make the directories in ConvE - saved_models, results, losses, logs, clusters
- For each model-dataset combination, we have included a bash script to train the original model, generate attacks from baselines and proposed attacks; and train poisoned model. These scripts are named as model-dataset.sh
- The instructions in these scripts are grouped together under the echo statements which indicate what they do.
- The hyperparameters in bash scripts are the ones used for the experiments reported in the submission
- The metrics on decoy triples can be computed by the script compute_decoy_metrics_WN18RR.sh or compute_decoy_metrics_FB15k-237.sh
- To reproduce the results, specific instructions from the bash scripts can be run on commandline or the full script can be run




