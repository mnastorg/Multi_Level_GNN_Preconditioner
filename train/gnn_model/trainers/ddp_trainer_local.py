########### PACKAGES ###################

import os
import sys
sys.path.append("..")

import numpy as np
import time
from tqdm import tqdm
from math import *
import json 

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities import utils
utils.set_seed()

import model

from utilities import utils

########################################

def setup_local(rank, world_size):

    # distributed variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def save_model(state, dirName = None, model_name = None):

    model_name = "{}.pt".format(model_name)
    save_path = os.path.join(dirName, model_name)
    path = open(save_path, mode="wb")
    torch.save(state, path)
    path.close()

def train(rank, world_size, dataset, file_config):
    
    json_config = open(file_config)
    config = json.load(json_config)
    json_config.close()
    
    config_model = config["config_model"]
    config_train = config["config_train"]

    init_epoch = 0

    # setup ddp
    setup_local(rank, world_size)

    # dataset
    train_dataset = dataset[0]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = config_train["batch_size"],
                              shuffle = False,
                              num_workers = 0,
                              sampler = train_sampler)

    val_dataset = dataset[1]
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size = config_train["batch_size"],
                              shuffle = False,
                              num_workers = 0,
                              sampler = val_sampler)

    # Build the model
    # Build the model
    DSSModel = model.DeepStatisticalSolver(config_model).to(rank)
        
    DSSModel = DistributedDataParallel(DSSModel, device_ids=[rank])

    if rank == 0 : 

        with open(os.path.join(config_train["path_logs"],"training_info.csv"), 'w') as f : 
            f.write(f"Number of GPU used : {torch.cuda.device_count()}")
            f.write("\n")
            f.write(f"{len(dataset[0])} Train samples, {len(dataset[1])} Val samples, {len(dataset[2])} test samples")
            f.write("\n")
            f.write(f"Batch size {config_train['batch_size']}")
            f.write("\n")
            f.write("Comment on the model : {}".format(config_train['comment']))
            f.write("\n")
            f.write(f"Number of parameters : {sum(p.numel() for p in DSSModel.parameters() if p.requires_grad)}")
            f.write("\n")
            f.write("\n")
            f.write("Complete Model : \n")
            f.write(str(DSSModel.module))
            f.write("}\n")
            f.close()

    optimizer = torch.optim.Adam(DSSModel.parameters(), lr = config_train["lr"])

    hist_train    = {"loss":[], "residual_loss":[], "mse_loss":[], "rel_loss":[]}
    hist_val      = {"loss":[], "residual_loss":[], "mse_loss":[], "rel_loss":[]}

    training_time = 0
    min_loss_save = config_train["min_loss_save"]

    for epoch in range(init_epoch, config_train["max_epochs"]):
        
        print(f"Epoch {epoch}")

        # time counter
        time_counter = time.time()

        # training loop
        cumul_loss, cumul_residual_loss,  cumul_mse_loss, cumul_rel_loss = 0, 0, 0, 0
        run_loss, run_residual_loss, run_mse_loss, run_rel_loss = 0, 0, 0, 0
        cumul = 0

        DSSModel.train()
        for i, train_batch in enumerate(train_loader):
            
            # zero the gradients
            optimizer.zero_grad()

            # put batch on gpu
            train_batch = train_batch.to(rank)

            # output of the model and losses
            U_sol, loss_dic = DSSModel(train_batch)
            loss =  loss_dic["train_loss"]

            # compute gradients
            loss.backward()
            
            # apply gradient clipping
            torch.nn.utils.clip_grad_norm_(DSSModel.parameters(), config_train["gradient_clip"])
            
            # optimizer step
            optimizer.step()
            
            # synchronize all gradients and make sure all weigths are updated
            dist.barrier()

            # train losses on rank 0
            if rank == 0 :

                # accumulate the losses
                cumul_loss += loss.item()
                cumul_residual_loss += loss_dic["residual_loss"][str(config_model["k"])].item()
                cumul_mse_loss += loss_dic["mse_loss"][str(config_model["k"])].item()
                cumul_rel_loss += loss_dic["rel_loss"][str(config_model["k"])].item()

                # Accumulate running loss and print every 25%
                run_loss += loss.item()
                run_residual_loss += loss_dic["residual_loss"][str(config_model["k"])].item()
                run_mse_loss += loss_dic["mse_loss"][str(config_model["k"])].item()
                run_rel_loss += loss_dic["rel_loss"][str(config_model["k"])].item()

                cumul += 1
                if i == ceil(0.25*len(train_loader)) or i == ceil(0.5*len(train_loader)) or i == ceil(0.75*len(train_loader)) : 
                    with open(os.path.join(config_train["path_logs"], 'train_metrics.csv'), 'a') as file : 
                        s = (f'\nEpoch {epoch}, {int(i * 100 / len(train_loader))}%'
                            f'\t Loss : {run_loss / cumul :.4e}'
                            f'\t RES : {run_residual_loss / cumul :.4e}'
                            f'\t MSE : {run_mse_loss / cumul :.4e}' 
                            f'\t REL : {run_rel_loss / cumul :.4e}')
                        file.write(s)
                        file.close()
                run_loss, run_residual_loss, run_mse_loss, run_rel_loss = 0, 0, 0, 0
                cumul = 0

        # end of current training epoch
        # print losses on rank 0 and perform validation step
        if rank == 0 :

            # save and print training losses
            hist_train["loss"].append(cumul_loss / len(train_loader))
            hist_train["residual_loss"].append(cumul_residual_loss / len(train_loader))
            hist_train["mse_loss"].append(cumul_mse_loss / len(train_loader))
            hist_train["rel_loss"].append(cumul_rel_loss / len(train_loader))
        
            with open(os.path.join(config_train["path_logs"],'train_metrics.csv'), 'a') as file :
                s = (f'\nTraining Epoch {epoch} :'   
                    f'\t Loss : {cumul_loss / len(train_loader) :.4e} '
                    f'\t RES : {cumul_residual_loss / len(train_loader) :.4e}'
                    f'\t MSE : {cumul_mse_loss / len(train_loader) :.4e}' 
                    f'\t REL : {cumul_rel_loss / len(train_loader) :.4e}') 
                file.write(s)
                file.close()        

            # validation loop 
            cumul_val_loss, cumul_val_residual_loss, cumul_val_mse_loss, cumul_val_rel_loss = 0, 0, 0, 0
            
            # eval mode
            DSSModel.eval()

            # deactivate gradients
            with torch.no_grad():
            
                for val_batch in val_loader:
            
                    val_batch = val_batch.to(rank)

                    U_sol, loss_dic = DSSModel(val_batch)
                    
                    loss =  loss_dic["train_loss"].mean()

                    cumul_val_loss += loss.item()
                    cumul_val_residual_loss += loss_dic["residual_loss"][str(config_model["k"])].item()
                    cumul_val_mse_loss +=  loss_dic["mse_loss"][str(config_model["k"])].item()
                    cumul_val_rel_loss +=  loss_dic["rel_loss"][str(config_model["k"])].item()

            hist_val["loss"].append(cumul_val_loss / len(val_loader))
            hist_val["residual_loss"].append(cumul_val_residual_loss / len(val_loader))
            hist_val["mse_loss"].append(cumul_val_mse_loss / len(val_loader))
            hist_val["rel_loss"].append(cumul_val_rel_loss / len(val_loader))
            
            with open(os.path.join(config_train["path_logs"],'train_metrics.csv'), 'a') as file : 
                s = (f'\nValidation Epoch {epoch} :'
                    f'\t Loss : {cumul_val_loss / len(val_loader) :.4e}'
                    f'\t RES : {cumul_val_residual_loss / len(val_loader) :.4e}' 
                    f'\t MSE : {cumul_val_mse_loss / len(val_loader) :.4e}' 
                    f'\t REL : {cumul_val_rel_loss / len(val_loader) :.4e}')
                file.write(s)
                file.close()

            # check validation finishes, now save model and time
            training_time = training_time + (time.time() - time_counter)

            # checkpoint current model
            checkpoint = {  'epoch'           : epoch,
                            'state_dict'      : DSSModel.module.state_dict(),                            
                            'opt'             : optimizer.state_dict(),
                            'gradient_clip'   : config_train["gradient_clip"],
                            'min_loss_save'   : min_loss_save,
                            'training_time'   : training_time,
                            'hist_train'      : hist_train,
                            'hist_val'        : hist_val,
                            }
            
            save_model(checkpoint, dirName = config_train["path_ckpt"], model_name = "running_model")

            # save checkpoint to best_model if residual validation loss is <= min loss save
            if hist_val["residual_loss"][-1] <= min_loss_save :
                save_model(checkpoint, dirName = config_train["path_ckpt"], model_name = "best_model")
                min_loss_save = hist_val["residual_loss"][-1]
                with open(os.path.join(config_train["path_logs"],'train_metrics.csv'), 'a') as file : 
                    s = (f'\nTraining Epoch {epoch} finished,' 
                        f'took current epoch {time.time() - time_counter:.2f}s,' 
                        f'cumulative time {training_time :.2f}s')
                    file.write(s)
                    file.write("\nMODEL SAVED")
                    file.close()

            else:
                with open(os.path.join(config_train["path_logs"],'train_metrics.csv'), 'a') as file : 
                    s = (f'\nTraining Epoch {epoch} finished,' 
                        f'took current epoch {time.time() - time_counter:.2f}s,' 
                        f'cumulative time {training_time :.2f}s')
                    file.write(s)
                    file.close()
                                  
        #save model
        save_model(checkpoint, dirName=config_train["path_ckpt"], model_name="final_model")

        dist.barrier()

    dist.destroy_process_group()  