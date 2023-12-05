import torch
import csv
import os

def csv_to_array(file_path, parsed_csv):
    """
    Args:
        file_path (_type_): _description_
        parsed_csv (_type_): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: _description_
    """
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
    

def tensor_from_file(path: str):
    """_summary_

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    data = []
    lines, cols = csv_to_array(path, data)
    data_tensor = torch.tensor(data)
    return data_tensor.view(lines, cols)
        