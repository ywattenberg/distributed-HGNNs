from scipy.io import mmread
import numpy as np
import torch
import time


# Load matrix from Matrix Market file
L = mmread('data/m_g_ms_gs/DVH.mtx')
R = mmread('data/m_g_ms_gs/invDE_HT_DVH.mtx')
features = mmread('data/m_g_ms_gs/features.mtx')
labels = mmread('data/m_g_ms_gs/labels.mtx')
print(f"L shape: {L.shape}")
print(f"R shape: {R.shape}")
print(f"features shape: {features.shape}")
print(f"labels shape: {labels.shape}")

# Convert to PyTorch tensors where L and R shall remain sparse
L = torch.sparse_coo_tensor(np.array(L.nonzero()), np.array(L.data), L.shape)
R = torch.sparse_coo_tensor(np.array(R.nonzero()), np.array(R.data), R.shape)
print(f"L shape: {L.shape}")
print(f"R shape: {R.shape}")

features = torch.tensor(features, dtype=torch.double)
labels = torch.tensor(labels, dtype=torch.double)

m, n_samples = 24622, 12311
n_features = 6144
labels = 40

# Trainable parameters
w = torch.ones([m], dtype=torch.double, requires_grad=True)
weights = torch.ones([n_features, labels], dtype=torch.double, requires_grad=True)
bias = torch.ones([labels], dtype=torch.float32, requires_grad=True)

t1 = time.time()
LwR = torch.mm(L, torch.mm(torch.sparse.spdiags(w, offsets=torch.tensor([0]), shape=(m,m)), R))
t2 = time.time()
print(f"LwR shape: {LwR.shape}")
print(f"Time taken: {t2-t1}")
XtB = torch.mm(features, weights) + bias
t3 = time.time()
print(f"XtB shape: {XtB.shape}")
print(f"Time taken: {t3-t2}")
G_3 = torch.mm(LwR, XtB)
t4 = time.time()
print(f"G_3 shape: {G_3.shape}")
print(f"Time taken: {t4-t3}")
G_4 = torch.nn.functional.relu(G_3)
t5 = time.time()
print(f"G_4 shape: {G_4.shape}")
print(f"Time taken: {t5-t4}")
print(f"Total time taken: {t5-t1}")








