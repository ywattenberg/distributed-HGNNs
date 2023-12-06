from typing_extensions import deprecated
import torch
import csv
import os
import pandas as pd

@deprecated("see :func:`tensor_from_file`")
def csv_to_array(file_path, parsed_csv):
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError(f"File {file_path} does not exist or is not a regular file")

    lines = 0

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            lines += 1
            parsed_row = []

            for cell in line:
                parsed_row.append(cell)

                try:
                    if isinstance(parsed_csv, list):
                        if issubclass(type(parsed_csv[0]), float):
                            parsed_csv.append(float(cell))
                        elif issubclass(type(parsed_csv[0]), int):
                            parsed_csv.append(int(cell))
                        elif issubclass(type(parsed_csv[0]), str):
                            parsed_csv.append(cell)
                except ValueError:
                    # non-parsable element in row
                    print("Element in row was not parseable")
                    parsed_row = []
                    break

    return lines, len(parsed_csv) // lines if lines > 0 else 0
    
@deprecated("use tensor_from_csv instead")
def tensor_from_file(path: str):
    """
    for training in python, use :func:`tensor_from_csv` instead
    """
    data = []
    lines, cols = csv_to_array(path, data)
    data_tensor = torch.tensor(data)
    return data_tensor.view(lines, cols)


def tensor_from_csv(path: str):
    # make sure the first row contains the column names only
    df = pd.read_csv(path, header=0)
    return torch.Tensor(df.values)

def coo_to_sparse(coo_tensor: torch.Tensor):
    # coo_list is a tensor of shape (n, 3) where n is the number of non-zero entries
    index = coo_tensor[:, :2].transpose(0, 1).to(torch.int64)
    values = coo_tensor[:, 2].squeeze()
    s = torch.sparse_coo_tensor(index, values)
    return s.to_dense()
    
