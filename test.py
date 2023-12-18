from scipy.io import mmread
import numpy as np
import torch
import time

labels = mmread('data/m_g_ms_gs/labels.mtx')
labels = torch.tensor(labels, dtype=torch.long).squeeze()
L = mmread('data/m_g_ms_gs/DVH.mtx')
R = mmread('data/m_g_ms_gs/invDE_HT_DVH.mtx')
L = torch.sparse_coo_tensor(np.array(L.nonzero()), np.array(L.data), L.shape)
R = torch.sparse_coo_tensor(np.array(R.nonzero()), np.array(R.data), R.shape)
print(f"L shape: {L.shape}")
print(f"R shape: {R.shape}")

m, n_samples = 24622, 12311
w = torch.ones([m], dtype=torch.double, requires_grad=True)
X = torch.rand([12311,128], dtype=torch.double, requires_grad=True)
B = torch.rand([40], dtype=torch.double, requires_grad=True)
T = torch.rand([128,40], dtype=torch.double, requires_grad=True)

wR = torch.mm(torch.eye(m)*w, R)
wR.retain_grad()
LwR = torch.mm(L, wR)
LwR.retain_grad()
print(f"LwR shape: {LwR.shape}")
G_2 = torch.mm(X, T) + B
G_3 = torch.mm(LwR, G_2)
G_2.retain_grad()
G_3.retain_grad()
G_4 = torch.nn.functional.relu(G_3)
G_4.retain_grad()
print(f"G_3 shape: {G_3.shape}")
print(f"labels shape: {labels.shape}")
loss = torch.nn.functional.cross_entropy(G_4, labels)
loss.backward(retain_graph=True)
relu_grad = G_3.clone()
relu_grad[relu_grad > 0] = 1
relu_grad[relu_grad <= 0] = 0
G_2_grad = torch.mm(LwR.t(), relu_grad * G_4.grad)
print(f"G_2_grad shape: {G_2_grad.shape}")
print(f"G_2 shape: {G_2.shape}")
print(torch.equal(G_2_grad, G_2.grad))
X_grad = torch.mm(G_2_grad, T.t())
print(f"X_grad shape: {X_grad.shape}")
print(f"X shape: {X.shape}")
print(torch.equal(X_grad, X.grad))
T_grad = torch.mm(X.t(), G_2_grad)
print(f"T_grad shape: {T_grad.shape}")
print(f"T shape: {T.shape}")
print(torch.equal(T_grad, T.grad))
w_grad = torch.mm( torch.mm(torch.mm(LwR, X), T), relu_grad *G_4.grad)
print(f"w_grad shape: {w_grad.shape}")
print(f"w shape: {w.shape}")
print(torch.equal(w_grad, w.grad))



exit()

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

loss = torch.nn.functional.cross_entropy(G_4, labels)
loss.backward(retain_graph=True)
prtin(f"detivative of loss w.r.t to g1" )
print(f"derivative of loss w.r.t. w: {w.grad.shape}")
print(f"derivative of loss w.r.t. weights: {weights.grad.shape}")
print(f"derivative of loss w.r.t. bias: {bias.grad.shape}")










