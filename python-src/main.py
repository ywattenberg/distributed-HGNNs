from mpi4py import MPI
import numpy as np
import torch

import yaml
import os
import argparse
import time

from variables import *
from utils import *
from model import DistModel

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
    loss_fn = torch.functional.F.cross_entropy
    
    # TODO: train model
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default=DEFAULT_SCRIPT_PATH)
    parser.add_argument('--data', help='path to data file', default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.full_load(f)
    
    distributed_learning(config=cfg)
    
    

if __name__ == "__main__":
    main()