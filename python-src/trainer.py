import mpi4py
mpi4py.rc.initialize = False 
mpi4py.rc.finalize = False 

from mpi4py import MPI
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from variables import MODELNET40_NUM_CLASSES
from utils.train_utils import CrossEntropyLoss

def train_torch_model(config, labels, features, model):
    # initialize parameters
    n_epochs = config["trainer"]["epochs"]
    stepsize_output = config["trainer"]["output_stepsize"]
    lr = config["trainer"]["learning_rate"]
    train_set_cutoff = config["data"]["test_idx"]
    
    lr_step_size = config["lr_scheduler"]["step_size"]
    lr_gamma = config["lr_scheduler"]["gamma"]
    
    train_labels = labels[:train_set_cutoff].long()
    test_labels = labels[train_set_cutoff:train_set_cutoff+labels.shape[0]].long()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) # TODO: add weight decay to config?
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    loss_fn = nn.CrossEntropyLoss() 
    
    # introduce metrics
    metric_accuracy = MulticlassAccuracy(num_classes=MODELNET40_NUM_CLASSES) 
    metric_f1_score = MulticlassF1Score(num_classes=MODELNET40_NUM_CLASSES)
    
    # training loop
    print("Starting training")
    model.train()
    
    for epoch in tqdm(range(n_epochs), desc="Training ... "):
        output = model(features)
        if model.fwd_only: # for timing only
            continue
        
        output_train = output[:train_set_cutoff]
        loss = loss_fn(output_train, train_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % stepsize_output == 0:
            with torch.no_grad():
                output_test = output[train_set_cutoff:train_set_cutoff+labels.shape[0]]
                test_loss = loss_fn(output_test, test_labels)
                test_acc = metric_accuracy(output_test, test_labels)
                test_f1 = metric_f1_score(output_test, test_labels)
                print(f"========== Epoch {epoch} ==========")
                print(f"Train loss: {loss} | Test loss: {test_loss}")
                print(f"Test accuracy: {test_acc} | Test F1 score: {test_f1}")    
    
    
def train_dist_model(config, labels, features, model):
    # initialize parameters
    n_epochs = config["trainer"]["epochs"]
    stepsize_output = config["trainer"]["output_stepsize"]
    lr = config["trainer"]["learning_rate"]
    train_set_cutoff = config["data"]["test_idx"]
    
    lr_step_size = config["lr_scheduler"]["step_size"]
    lr_gamma = config["lr_scheduler"]["gamma"]
    
    train_labels = labels[:train_set_cutoff].long()
    test_labels = labels[train_set_cutoff:train_set_cutoff+labels.shape[0]].long()
    
    loss_fn = CrossEntropyLoss
    
    # initialize MPI
    MPI.Init()
    
    #TODO: scatter data
    
    # TODO: train model and convert data type to numpy correctly
    for epoch in tqdm(range(n_epochs), desc="Training ... "):
        output = model(features)
        if model.fwd_only: # for timing only
            continue
        
        output_train = output[:train_set_cutoff]
        # print(output_train.shape)
        # print(train_labels)
        loss = loss_fn(output_train, train_labels)
        
    """
        model.backward(loss)
        model.update(lr)
    """
        
        