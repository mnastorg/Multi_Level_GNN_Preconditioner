### Packages ###

import os
import sys
import json 
import shutil
import argparse
import warnings 
warnings.simplefilter('ignore')

from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utilities import reader

from utilities import utils
from tqdm import tqdm

from trainers import ddp_trainer_local
from trainers import ddp_trainer_slurm
from trainers import dp_trainer

from utilities import utils
utils.set_seed()

#################

def get_slurm_info():

    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

    return world_size, rank, local_rank

def generate_results_folder(name):
    
    results_folder = "results"
    path_results = os.path.join(results_folder, name)
    path_logs = os.path.join(path_results, "logs")
    path_ckpt = os.path.join(path_results, "ckpt")

    # build folder to save results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    if not os.path.exists(path_results):
        os.makedirs(path_results, exist_ok=True)
    
    if not os.path.exists(path_logs):
        os.makedirs(path_logs, exist_ok=True)

    if not os.path.exists(path_ckpt):
        os.makedirs(path_ckpt, exist_ok=True)

    return path_results, path_logs, path_ckpt

def update_config_file(args, path_logs, path_ckpt):

    shutil.copy(args.config_file, path_logs)

    open_config = open(os.path.join(path_logs, args.config_file))
    config = json.load(open_config)
    open_config.close()
    
    config["config_train"]["path_logs"] = path_logs
    config["config_train"]["path_ckpt"] = path_ckpt

    json_object = json.dumps(config, indent=2)
    with open(os.path.join(path_logs, args.config_file), "w") as outfile:
        outfile.write(json_object)
    
    return config

def generate_dataset(config):

    path_dataset = config["config_train"]["path_dataset"]

    dataset_train   = reader.BuildDataset(root = path_dataset, mode = 'train'  , precision = torch.float)
    dataset_val     = reader.BuildDataset(root = path_dataset, mode = 'val'    , precision = torch.float)
    dataset_test    = reader.BuildDataset(root = path_dataset, mode = 'test'   , precision = torch.float)

    return [dataset_train, dataset_test, dataset_val]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description  =  'Test Deep Statistical Solvers')
    parser.add_argument("--training_mode",  type = str,   default = "ddp_slurm",    help = "Choose between dp, ddp_slurm, ddp_local")
    parser.add_argument("--config_file",    type = str,   default = "config.json",  help = "Config files")
    parser.add_argument("--results_name",   type = str,   default = "temp",         help = "Name of the results folder")

    args = parser.parse_args()

    if args.training_mode == "ddp_slurm":
        
        world_size, rank, local_rank = get_slurm_info()
        print(f"World Size {world_size}, Rank {rank}, Local rank {local_rank}")

        # setup DDP
        dist.init_process_group('nccl', world_size=world_size, rank=rank)

        if rank == 0:
            
            print(f"Rank {rank} is generating the dataset and updating the config file")

            path_results, path_logs, path_ckpt = generate_results_folder(args.results_name)

            config = update_config_file(args, path_logs, path_ckpt)

            if not os.path.exists(os.path.join(config["config_train"]["path_dataset"], "processed_dss")):
                dataset = generate_dataset(config)

            shutil.copy(os.path.join(config["config_train"]["path_dataset"], "data/normalization_info.json"), path_logs)

        dist.barrier()

        path_logs = os.path.join("results", args.results_name, "logs")

        open_config = open(os.path.join(path_logs, args.config_file))
        config = json.load(open_config)
        open_config.close()

        dataset = generate_dataset(config)
        
        ddp_trainer_slurm.train(world_size, 
                                rank, 
                                local_rank,
                                dataset, 
                                os.path.join(path_logs, args.config_file))
    
    if args.training_mode == "ddp_local":

        world_size = torch.cuda.device_count()

        path_results, path_logs, path_ckpt = generate_results_folder(args.results_name)

        config = update_config_file(args, path_logs, path_ckpt)
        
        dataset = generate_dataset(config)

        shutil.copy(os.path.join(config["config_train"]["path_dataset"], "data/normalization_info.json"), path_logs)

        mp.spawn(ddp_trainer_local.train, 
                args = (world_size, dataset, os.path.join(path_logs, args.config_file)), 
                nprocs=world_size, 
                join=True)

    if args.training_mode == "dp":
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path_results, path_logs, path_ckpt = generate_results_folder(args.results_name)

        config = update_config_file(args, path_logs, path_ckpt)
        
        dataset = generate_dataset(config)

        shutil.copy(os.path.join(config["config_train"]["path_dataset"], "data/normalization_info.json"), path_logs)

        dp_trainer.train(device, dataset, os.path.join(path_logs, args.config_file))

