"""
This script provides a testing ground for the inspection of the sparsity of the
matrices used in the hypergraph convolution operation
"""

import torch
import torch.nn as nn

# feel free to tune these parameters
V = 500
E = 1000
MAX_DEG = 20
N_FEATURES = 200
EDGE_PROB = 0.0005

# weight and adjacency and parameter matrices
W = nn.Parameter(torch.eye(E, E)) 
H = torch.bernoulli(torch.full((V, E), EDGE_PROB))
THETA = torch.rand(N_FEATURES, N_FEATURES)

# diagonal degree matrices
D_v = torch.diag(torch.randint(0, MAX_DEG, (V,))).float()
D_e = torch.diag(torch.randint(0, MAX_DEG, (E,))).float()

# input matrix
X = torch.rand(V, N_FEATURES)

# print mm results
print("D_v: ", D_v)
print("D_e: ", D_e)
print("H: ", H)
print("L: ", D_v @ H)                                # sparse
print("R: ", D_e @ H.t() @ D_v)                      # sparse

LWR = D_v @ H @ W @ D_e @ H.t() @ D_v                # sparse
LWRX =  LWR @ X                                      # sparse times dense (SpMM)
Y = LWRX @ THETA                                     # sprase/dense times dense 
print(f"LWR: shape:{LWR.shape}, {LWR.nonzero().shape[0]}/{LWR.shape[0]*LWR.shape[1]} non-zeros")      # sparse
print(f"LWRX: shape:{LWRX.shape}, {LWRX.nonzero().shape[0]}/{LWRX.shape[0]*LWRX.shape[1]} non-zeros")  # sparse if...?
print(f"Y: shape:{Y.shape}, {Y.nonzero().shape[0]}/{Y.shape[0]*Y.shape[1]} non-zeros")   # nnz(Y) = nnz(LWRX)