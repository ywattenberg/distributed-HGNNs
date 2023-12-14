import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False 

from mpi4py import MPI
import numpy as np

def CrossEntropyLoss(pred: np.ndarray, target: np.ndarray, n_samples: int, sum=True):
    """
    Implements Cross entropy loss function for distributed training.
    This function does not use sparse format, but wraps around numpy ndarrays instead.
    
    n_samples provides a workaround to access the total number of rows and columns
    of the pred matrix not in a sparse format (TODO: get n_samples by MPI allreduce)
    
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        n_samples (int): number of total samples in the batch (before scatter)
        sum (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(pred.shape) == 1:
        local_rows, local_cols = 1, pred.shape[0]
        pred = np.expand_dims(pred, axis=0)
    else:
        local_rows, local_cols = pred.shape

    # Calculate the max value for each row
    max_val = np.max(pred, axis=1)
    max_val_global = comm.allreduce(max_val, op=MPI.MAX) # max_val[0]

    # Calculate local_sum 
    local_sum = np.zeros(local_rows)
    for i in range(local_rows):
        for j in range(local_cols):
            local_sum[i] += np.exp(pred[i, j] - max_val[i])
            
    local_sum = comm.allreduce(local_sum, op=MPI.SUM) # local_sum[0]
    
    for i in range(local_rows):
        local_sum[i] = max_val[i] + np.log(local_sum[i])

    # Calculate x_y_n locally for the current process
    samples_per_proc = n_samples // size  
    row_offset = rank * samples_per_proc
    x_y_n_local = np.zeros(local_rows)
    if row_offset <= target.max() < row_offset + samples_per_proc:
        # Determine which rows this process is responsible for
        local_row_indices = target - row_offset
        mask = (local_row_indices >= 0) & (local_row_indices < local_rows)
        x_y_n_local[mask] = pred[local_row_indices[mask], target[mask]]

    #  get x_y_n across all processes
    x_y_n = np.zeros(local_rows)
    comm.Allreduce(x_y_n_local, x_y_n, op=MPI.SUM)

    total_rows = float(pred.shape[0])
    loss_local = np.sum((-x_y_n + local_sum) / (1.0 if sum else total_rows))
    loss = comm.allreduce(loss_local, op=MPI.SUM) / size

    return loss


def DerivativeCrossEntropyLoss(pred, target):
    """
    Compute the derivative of the CrossEntropy Loss

    Args:
        pred (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(pred.shape) == 1:
        local_rows, local_cols = 1, pred.shape[0]
        pred = np.expand_dims(pred, axis=0)
    else:
        local_rows, local_cols = pred.shape

    # Calculate local_sum
    local_sum = np.sum(np.exp(pred), axis=1)
    local_sum = comm.allreduce(local_sum, op=MPI.SUM)

    # Calculate derivative
    n, c = comm.allreduce(np.array([pred.shape[0]]), op=MPI.SUM)
    samples_per_processor = n // size
    row_offset = rank * samples_per_processor
    derivative_out = np.zeros((local_rows, local_cols))
    
    for i in range(local_rows):
        for j in range(local_cols):
            if row_offset + i < n: 
                target_idx = row_offset + i
                if j == target[target_idx]:
                    derivative_out[i, j] = np.exp(pred[i, j]) / local_sum[i] - 1.0
                else:
                    derivative_out[i, j] = np.exp(pred[i, j]) / local_sum[i]

    return derivative_out