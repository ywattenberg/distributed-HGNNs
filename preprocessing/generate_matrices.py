import os
import numpy as np
import pandas as pd
from io import BytesIO
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

from utils import hypergraph_utils as hgut
from config import get_config
from datasets import load_feature_construct_H

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

def get_filename():
    filename = ""
    if cfg['use_mvcnn_feature']:
        filename += "m"
    if cfg['use_gvcnn_feature']:
        if filename == "":
            filename += "g"
        else:
            filename += "_g"
    if cfg['use_mvcnn_feature_for_structure']:
        if filename == "":
            filename += "ms"
        else:
            filename += "_ms"
    if cfg['use_gvcnn_feature_for_structure']:
        if filename == "":
            filename += "gs"
        else:
            filename += "_gs"
    return filename

def generate_matrices(filename, variable_weight=True):
    # initialize data
    data_dir = cfg['modelnet40_ft'] if cfg['on_dataset'] == 'ModelNet40' \
        else cfg['ntu2012_ft']
    fts, lbls, idx_train, idx_test, H = \
        load_feature_construct_H(data_dir,
                                m_prob=cfg['m_prob'],
                                K_neigs=cfg['K_neigs'],
                                is_probH=cfg['is_probH'],
                                use_mvcnn_feature=cfg['use_mvcnn_feature'],
                                use_gvcnn_feature=cfg['use_gvcnn_feature'],
                                use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
                                use_gvcnn_feature_for_structure=cfg['use_gvcnn_feature_for_structure'])        
        
    G = hgut.generate_G_from_H(H, variable_weight=variable_weight)
    
    if not os.path.exists(f"data/{filename}"):
        os.makedirs(f"data/{filename}")
    
    np.save(f"data/{filename}/features.npy", fts)
    np.save(f"data/{filename}/labels.npy", lbls.astype(int))
    np.save(f"data/{filename}/idx_train.npy", idx_train)
    np.save(f"data/{filename}/idx_test.npy", idx_test)
    np.save(f"data/{filename}/H.npy", H)
    
    if variable_weight:
        DVH, W, invDE_HT_DVH = G

        # write matrices to file
        np.save(f"data/{filename}/DVH.npy", DVH)
        np.save(f"data/{filename}/invDE_HT_DVH.npy", invDE_HT_DVH) 
    else:
        np.save(f"data/{filename}/G.npy", G)

def convert_to_mm(filename, variable_weight=True):
    fts = np.load(f"data/{filename}/features.npy")
    mmwrite(f"data/{filename}/features.mtx", fts)
    lbls = np.load(f"data/{filename}/labels.npy")
    mmwrite(f"data/{filename}/labels.mtx", lbls)
    
    if variable_weight:
        DVH = np.load(f"data/{filename}/DVH.npy")
        mmwrite(f"data/{filename}/DVH.mtx", coo_matrix(DVH))
        
        invDE_HT_DVH = np.load(f"data/{filename}/invDE_HT_DVH.npy")
        mmwrite(f"data/{filename}/invDE_HT_DVH.mtx", coo_matrix(invDE_HT_DVH))
    else:
        G = np.load(f"data/{filename}/G.npy")
        mmwrite(f"data/{filename}/G.mtx", coo_matrix(G))
    
def convert_matrices(filename, variable_weight=True):
    
    fts = np.load(f"data/{filename}/features.npy")
    df = pd.DataFrame(fts)
    df.to_csv(f"data/{filename}/features.csv", index=False, header=False)
    
    lbls = np.load(f"data/{filename}/labels.npy")
    df = pd.DataFrame(lbls)
    df.to_csv(f"data/{filename}/labels.csv", index=False, header=False)
    
    if variable_weight:
        DVH = np.load(f"data/{filename}/DVH.npy")
        indexes = np.nonzero(DVH)
        with open(f"data/{filename}/DVH_coo.csv", "w") as f:
            f.write("row,col,val\n")
            for (i,j) in zip(*indexes):
                f.write(f"{i},{j},{DVH[i,j]}\n")
        
        invDE_HT_DVH = np.load(f"data/{filename}/invDE_HT_DVH.npy")
        indexes = np.nonzero(invDE_HT_DVH)
        with open(f"data/{filename}/invDE_HT_DVH_coo.csv", "w") as f:
            f.write("row,col,val\n")
            for (i,j) in zip(*indexes):
                f.write(f"{i},{j},{invDE_HT_DVH[i,j]}\n")
    else:
        G = np.load(f"data/{filename}/G.npy")
        indexes = np.nonzero(G)
        with open(f"data/{filename}/G_coo.csv", "w") as f:
            f.write("row,col,val\n")
            for (i,j) in zip(*indexes):
                f.write(f"{i},{j},{G[i,j]}\n")
  
if __name__ == "__main__":
    filename = get_filename()
    # generate_matrices(filename, variable_weight=False)
    convert_to_mm(filename, variable_weight=False)
    convert_matrices(filename, variable_weight=False)