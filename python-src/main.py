from mpi4py import MPI
import numpy as np
import torch

import yaml
import os
import argparse
import time

from variables import *
from utils import *
from model import DistModel, TorchModel
from trainer import train_model


"""
TODO:
- implement distributed learning
- implement parallel matrix formats 
- test matrix_utils.py
"""

# example
def matvec(comm, A, x):
    m = A.shape[0] # local rows
    p = comm.Get_size()
    xg = np.zeros((m, p), dtype='d')
    comm.Allgather([x,  MPI.DOUBLE],
                   [xg, MPI.DOUBLE])
    y = np.dot(A, xg)
    return y

def distributed_learning(config):
    labels = tensor_from_file(config["data"]["labels_path"])
    print(labels.shape)
    
    features = tensor_from_file(config["data"]["features_path"])
    f_cols = features.shape[1]
    
    model = DistModel(config, f_cols)
    loss_fn = torch.nn.functional.cross_entropy
    
    # TODO: train model
    
def torch_training(config):
    """
    Model stats (paper config):
    500 EPOCHS:
        Train loss: 0.27136990427970886 | Test loss: 0.42119860649108887
        Test accuracy: 0.8523455262184143 | Test F1 score: 0.8495508432388306
        
    1000 EPOCHS:
        Train loss: 0.18341749906539917 | Test loss: 0.3153392970561981
        Test accuracy: 0.8947528004646301 | Test F1 score: 0.8895837664604187
    """
    coo_list = tensor_from_csv(config["data"]["g_path"])
    left_side = coo_to_sparse(coo_list) # 12311 x 12311
    
    labels = tensor_from_csv(config["data"]["labels_path"]) # 12311 x 1
    labels = labels.squeeze(-1)
    
    features = tensor_from_csv(config["data"]["features_path"]) # 12311 x 6145
    f_cols = features.shape[1]

    model = TorchModel(config=config, in_dim=f_cols, left_side=left_side)
    
    # TODO: timing 
    train_model(config, labels, features, model)
    
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default=DEFAULT_SCRIPT_PATH)
    parser.add_argument('--data', help='path to data file', default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.full_load(f)
    
    torch_training(cfg)
    
    

if __name__ == "__main__":
    main()