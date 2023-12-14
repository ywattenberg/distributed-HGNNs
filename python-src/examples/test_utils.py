import sys
sys.path.append('.')

# import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import *
from utils.data_utils import *
from utils.matrix_utils import *

def test_simple_dist_mm():
    """
    Tests the correctness of the distributed matrix multiplication
    """
    # numpy direct matrix multiplication
    test_mat_0 = np.array([[1., 2, 3, 4], # matrix
                           [4, 5, 6, 7]])
    # test_mat_1 = np.array([1., 2, 3, 4])  # vector
    test_mat_rhs = np.eye(4)
    res_direct = test_mat_0 @ test_mat_rhs
    
    # numpy distributed matrix multiplication
    import mpi4py
    mpi4py.rc.initialize = False   
    mpi4py.rc.finalize = False 
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        mat_l = test_mat_0
        mat_r = test_mat_rhs
    else:
        mat_l, mat_r = None, None

    mat_l = comm.scatter(mat_l, root=0)
    mat_r = comm.scatter(mat_r, root=0)
    res_dist = simple_dist_mm(mat_l, mat_r, comm)
    MPI.Finalize()
    
    # compare
    if rank == 0:
        eps = 1e-5
        print("res direct multiplication: ", res_direct)
        print("res distributed mm with np arrays: ", res_dist)
        assert np.allclose(res_direct, res_dist, rtol=eps), "solution is not correct"


def test_ce_loss():
    """
    Tests the  correctness of the distributed 
    cross-entropy loss function
    """
    # torch CE Loss
    test_pred = torch.tensor([[1., 2, 3], 
                              [4, 5, 6]])
    test_target = torch.tensor([0, 1])
    loss_torch = nn.CrossEntropyLoss()(test_pred, test_target)
    
    # numpy distributed CE Loss
    import mpi4py
    mpi4py.rc.initialize = False   
    mpi4py.rc.finalize = False 
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        pred = np.array([[1., 2, 3], 
                        [4, 5, 6]])
        target = np.array([[0], [1]])
    else:
        pred, target = None, None

    pred = comm.scatter(pred, root=0)
    target = comm.scatter(target, root=0)
    loss_np = CrossEntropyLoss(pred, target, n_samples=2)
    MPI.Finalize()
    
    # compare
    if rank == 0:
        eps = 1e-5
        print("ce loss torch: ", loss_torch.item())
        print("ce loss np: ", loss_np)
        assert abs(loss_torch.item() - loss_np) < eps, "solution is not correct"
    
if __name__ == '__main__':
    # assume pred is a logit matrix, not probabilities
    # n_classes = 3, n_samples = 2
   test_ce_loss()