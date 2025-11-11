# NPN_probing repository
This repository contains all the code, data, and output files for the ConLL Submission: "Construction Identification and Disambiguation Using BERT:
A Case Study of NPN"

# How to run
The main script for these experiments is the get_embeddings.py script. The other scripts are just for preprocessing, and aren't necessary to run anymore. The cleaned data is already in the repo. The bash script "run_few_shot.sh" can be used to run multiple seeds and data settings at once. 

# Example arguments
python3 get_embeddings.py -d ./data/raw_NPN_data_cleaned_subtype.tsv -i ./data/train_test_split_train_balanced_Y.json --semantic

# Data folder
clean6.tsv is the main dataset with all annotations. Other data files store perturbed versions of the dataset (these are automatically generated). There is also a json file storing the indices used for experiments. This is because only a subset of the entire dataset is used for our experiments (see the paper for a discussion for why we excluded some overly frequent lemma examples). This file must be supplied to get_embeddings.py with the flag --index_file.

#GloVe
This repo expects the "glove.6B.300d.txt" file to be located in a subdirectory "glove" in the main repo directory. We don't include the file because it's too big, but it's freely available on the web.

# Outputs folder
All outputs for the paper are in the outputs folder. Visualizations were generated using the attached ipynb notebook. 
