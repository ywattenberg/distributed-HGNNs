import math

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Classes for custom distributed learning 
"""
class DistConv:
    def __init__(self, in_dim: int, out_dim: int, with_bias: bool = False) -> None:
        # TODO: init
        pass
    
    def forward(self):
        pass

class DistModel:
    def __init__(self, config: dict, in_dim: int) -> None:
        self.input_dim = in_dim
        self.output_dim = config["model_properties"]["classes"]
        self.dropout = config["model_properties"]["dropout_rate"]
        self.lay_dim = config["model_properties"]["hidden_dims"]
        self.number_of_hid_layers = len(self.lay_dim)
        self.with_bias = config["model_properties"]["with_bias"]
        
        self.layers = []
        
        if self.number_of_hid_layers > 0:
            in_conv = DistConv(self.input_dim, self.lay_dim[0], self.with_bias)
            self.layers.append(in_conv)
            for i in range(1, self.number_of_hid_layers):
                self.layers.append(DistConv(self.lay_dim[i-1], self.lay_dim[i], self.with_bias))
            out_conv = DistConv(self.lay_dim[-1], self.output_dim, self.with_bias)
            self.layers.append(out_conv)
        else:
            out_conv = DistConv(self.input_dim, self.output_dim, self.with_bias)
            self.layers.append(out_conv)
       
       
    def forward(self):
        pass 
    
    
"""
Class for PyTorch learning
"""
class TorchHypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, with_bias: bool = False, t: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.__reset_params()
            
    def forward(self, x):
        return torch.mm(x, self.weights) + self.bias if self.bias is not None else torch.mm(x, self.weights)

    def __reset_params(self):
        stdv = 1./math.sqrt(self.weights.shape[1])
        nn.init.uniform_(self.weights, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)


class TorchModel(nn.Module):
    def __init__(self, config: dict, in_dim: int, left_side: torch.Tensor) -> None:
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = config["model"]["classes"]
        self.dropout = config["model"]["dropout_rate"]
        self.lay_dim = config["model"]["hidden_dims"]
        self.number_of_hid_layers = len(self.lay_dim)
        self.with_bias = config["model"]["with_bias"]
        self.left_side = left_side
        
        self.layers = nn.ModuleList()
        
        if self.number_of_hid_layers > 0:
            in_conv = TorchHypergraphConv(self.input_dim, self.lay_dim[0], self.with_bias)
            self.layers.append(in_conv)
            for i in range(1, self.number_of_hid_layers):
                self.layers.append(TorchHypergraphConv(self.lay_dim[i-1], self.lay_dim[i], self.with_bias))
            out_conv = TorchHypergraphConv(self.lay_dim[-1], self.output_dim, self.with_bias)
            self.layers.append(out_conv)
        else:
            out_conv = TorchHypergraphConv(self.input_dim, self.output_dim, self.with_bias)
            self.layers.append(out_conv)
       
       
    def forward(self, x):
        x = self.layers[0](x)
        x = torch.mm(self.left_side, x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        for i in range(1, len(self.layers)-1):
            l = self.layers[i]
            x = F.relu(l(x))
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = self.layers[-1](x)
        return x