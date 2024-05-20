#!/bin/bash

source activate dss_env

python3 dataset_creation.py \
--path_dataset dTest \ 
--nb_problems 5 \
--nb_nodes_subdomains 1000 \

exit 0
