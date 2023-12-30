import torch

DEFAULT_SCRIPT_PATH = "../config/paper_config.yaml"
DEFAULT_DATA_PATH = "../data"
MODELNET40_NUM_CLASSES = 40
GPLUS_NUM_CLASSES = 469 # 468 + 1 -> extra class for isolated vertices

FEATURE_TYPE = torch.float
LABEL_TYPE = torch.long
DEVICE = "CPU" 