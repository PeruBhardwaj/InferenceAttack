<h1 align="left">
  Code Structure
</h1>
<h3 align="left">This file describes the structure of the code</h3>

Commandline instructions for all experiments are available in bash scripts at this level
 
The main codebase is in `ConvE`
- script to preprocess data (generate dictionaries) is `preprocess.py`
- script to generate evaluation filters and training tuples is `wrangle_KG.py`
- script to train a KGE model is `main.py`
- script to select target triples from the test set is `select_targets.py`
- Random neighbourhood baseline is in `rand_add_attack_1.py`
- Random global baseline is in `rand_add_attack_2.py`
- Zhang et al. baseline is implemented in `ijcai_add_attack.py`
- CRIAGE baseline is in `criage_add_attack_1.py`
- Proposed symmetric attacks in `sym_add_attack_{1,2,3}`
    - 1 for soft truth score 
    - 2 for KGE ranks 
    - 3 for cosine distance
- Proposed inversion attacks in `inv_add_attack_{1,2,3}`
    - 1 for soft truth score 
    - 2 for KGE ranks 
    - 3 for cosine distance
- Proposed composition attacks in `com_add_attack_{1,2,3}`
    - 1 for soft truth score 
    - 2 for KGE ranks 
    - 3 for cosine distance
- Elbow method to select clusters is in `clustering_elbow.ipynb`
- Script to generate clusters is `create_clusters.py`
- Script to compute metrics on decoy set in `decoy_test.py`
- Folder `elbow_plots` contains the elbow plots
- Folder `data` will contain datasets generated from running the experiments. 
    - These are named as `attack_model_dataset_split_budget_run` 
    - here `split=1` for target split, `budget=1` for most attacks except random global with 2 edits, and `run` is the number for a random run
    - For Zhang et al. attacks, an additional argument is down sampling percent
- Folder `saved_models`, `clusters`, `logs`, `results` and `losses` are also empty but will be used if a script is run
