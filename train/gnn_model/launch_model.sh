#!/bin/bash

source activate dss_env

backup_dir=$(date +'%d_%m_%Y_%T')

python3 main.py --training_mode dp --config_file config.json --results_name ${backup_dir}

exit 0
