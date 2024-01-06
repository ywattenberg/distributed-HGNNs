import sys
sys.path.append("..")

import numpy as np
import os
import csv
import pandas as pd
import time
import psutil

from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite
# from ..variables import GPLUS_NUM_CLASSES

NUM_CPUS = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
 
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
 
        return result
    return wrapper



def generate_gplus(path="data/google+", reset=False):
    """
    Given an user with the user properties, predict which circle the user is in
    """
    
    # get node ids
    if reset or not os.path.exists(f"{path}/nid.txt"):
        edges = pd.read_csv(f"{path}/gplus_combined.txt", delim_whitespace=True, header=None)#np.loadtxt(f"{path}/gplus_combined.txt", max_rows=1000, dtype=str)
        print("READ EDGES DONE")
        
        node_ids = np.unique(edges)
        print("EXTRACT NODE IDS DONE")
 
        np.savetxt(f"{path}/nid.txt", node_ids, fmt='%s', delimiter=' ')
        print("SAVE NODE IDS DONE")
    else:
        node_ids = np.loadtxt(f"{path}/nid.txt", dtype=str)
        
    # get the adj and the feature matrix
    # adj matrix is of shape (n_nodes, n_circles)
    # where one hyperedge represents one cricle
    files = os.listdir(f"{path}/gplus")
    circle_id_files = [c[:-len(".circles")] for c in files if c.endswith("circles")] # 132
    
    nid2gid = {}
    gname2gid = {} 
    n_groups = 0 # 468
    
    # for each circle, extract the ids of the nodes in the circle and set the corresponding entries in the hypergraph adj matrix to 1
    H = np.zeros((len(node_ids), 468), dtype=int)
    nid2idx = {nid:idx for idx, nid in enumerate(node_ids)}
    
    for f in circle_id_files:
        if os.path.getsize(f"{path}/gplus/{f}.circles") == 0:
            continue
        
        node_ids_in_circle = pd.read_csv(f"{path}/gplus/{f}.circles", dtype=str, header=None) # groups
        for i in range(len(node_ids_in_circle)):
            group_name = node_ids_in_circle.iloc[i].str.split("\t")[0][0]
            node_ids_in_group = node_ids_in_circle.iloc[i].str.split("\t")[0][1:]
            
            # map nodes to groups and groups to group ids
            if group_name not in gname2gid:
                gname2gid[group_name] = n_groups
                n_groups += 1
            for nid in node_ids_in_group:
                if nid2gid.get(nid2idx[nid]) is None:
                    nid2gid[nid2idx[nid]] = gname2gid.get(group_name)

            # map nodes to hyperedges
            # print(gname2gid.get(group_name))
            H[np.array([nid2idx[nid] for nid in node_ids_in_group]), gname2gid.get(group_name)] = 1
                
    if reset or not os.path.exists(f"{path}/H_coo.npy"):    
        H = coo_matrix(H) # nnz = 3114012
        np.save(f"{path}/H_coo.npy", H)
        print("SAVED ADJACENCY MATRIX")
    
    # get labels
    if reset or not os.path.exists(f"{path}/labels.npy"):
        labels = np.array(list(nid2gid.items()))
        np.save(f"{path}/labels.npy", labels)
        print("SAVED LABELS")
    
    # get features
    fn2fid = {}
    n_feats = 0 # 19044
    for f in circle_id_files:
        if os.path.getsize(f"{path}/gplus/{f}.featnames") == 0:
            continue
        
        with open(f"{path}/gplus/{f}.featnames", "r") as f:
            for line in f:
                fn = line.split("\t")[0].split(' ', 1)[1][:-1]
                if fn not in fn2fid:
                    fn2fid[fn] = n_feats
                    n_feats += 1
                    
    fn2fid = np.array(list(fn2fid.items()))
    np.savetxt(f"{path}/fn2fid.txt", fn2fid, fmt='%s', delimiter='$$')        
    fn2fid = {k: int(v) for k, v in dict(fn2fid).items()}
        
    # fill the feature matrix
    X = np.zeros((len(node_ids), n_feats), dtype=int)
    for file in circle_id_files:
        if os.path.getsize(f"{path}/gplus/{file}.feat") == 0:
            continue
        
        with open(f"{path}/gplus/{file}.feat", "r") as f, open(f"{path}/gplus/{file}.featnames", "r") as g:
            curr_featnames = []
            for line in g:
                fn = line.split("\t")[0].split(' ', 1)[1][:-1]
                if fn not in curr_featnames:
                        curr_featnames.append(fn)

            for line in f:
                node_feats = list(line.split("\t")[0].split(' '))
                nid = nid2idx[node_feats[0]]
                for feat_idx in range(len(node_feats[1:])):
                    if node_feats[feat_idx+1] == '1':
                        feat_name = curr_featnames[feat_idx]
                        fid = fn2fid[feat_name]
                        X[nid][fid] = 1
                            
    features = coo_matrix(X)
    if reset or not os.path.exists(f"{path}/features_coo.npy"):
        np.save(f"{path}/features_coo.npy", features)
        print("SAVED FEATURES")
        

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
                

@profile          
def generate_mats_from_H(H, variable_weight=False, target_save_dir="google+"):
    """
    calculate G from hypgraph incidence matrix H 
    :param H: hypergraph incidence matrix H 
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) == coo_matrix:
        H = H.toarray()
    else:
        H = np.array(H)
    H = H.astype(float)
    # H = H[:1000, :] # test truncation
    
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    
    def _pow_ignore_zero(x, power):
        return np.where(x == 0, 0, np.power(x, power))
    
    print("COMPUTING LHS")
    invDE = np.mat(np.diag(_pow_ignore_zero(DE, -1))) #np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(_pow_ignore_zero(DV, -0.5)))# np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    
    print("mm chain 1: ", DV2.shape, H.shape)
    print("mm chain 2: ", invDE.shape, HT.shape, DV2.shape)
    print("mm chain 3: ", DV2.shape, H.shape, W.shape, invDE.shape, HT.shape, DV2.shape)

    print("COMPUTING CONV")
    if variable_weight:
        start1 = time.time()
        DV2_H = coo_matrix(DV2) * coo_matrix(H)
        end1 = time.time()
        print(f"chain 1 took {end1-start1} seconds")
        
        start2 = time.time()
        invDE_HT_DV2 = invDE * (coo_matrix(HT)) * DV2
        end2 = time.time()
        print(f"chain 2 took {end2-start2} seconds")
        
        return DV2_H, W, invDE_HT_DV2
    else:
        start3 = time.time()
        L = coo_matrix(DV2) * coo_matrix(H)
        R = invDE * coo_matrix(HT) * DV2
        print("computing L and R finished")
        G = L * R # assume W is identity
        end3 = time.time()
        print(f"chain3 took {end3-start3} seconds")
        return G
      
    
def convert_matrices_to_mm(filename, variable_weight=True):
    lbls = np.load(f"data/{filename}/labels.npy")
    fts = np.load(f"data/{filename}/features_coo.npy", allow_pickle=True).item()
    mmwrite(f"data/{filename}/labels.mtx", lbls)
    mmwrite(f"data/{filename}/features.mtx", fts)
        
    if variable_weight:
        DVH = np.load(f"data/{filename}/DVH.npy", allow_pickle=True).item()
        mmwrite(f"data/{filename}/DVH.mtx", coo_matrix(DVH))
        
        invDE_HT_DVH = np.load(f"data/{filename}/invDE_HT_DVH.npy")#.item()
        mmwrite(f"data/{filename}/invDE_HT_DVH.mtx", coo_matrix(invDE_HT_DVH))
    else:
        G = np.load(f"data/{filename}/G.npy")
        mmwrite(f"data/{filename}/G.mtx", coo_matrix(G))          
            
               
def generate_gplus_matrices(reset, variable_weight=True):   
    dataset_name = "google+"
    
    if not os.path.exists(f"data/{dataset_name}"):
        os.makedirs(f"data/{dataset_name}")
    
    H = np.load(f"data/google+/H_coo.npy", allow_pickle=True).item()
    G = generate_mats_from_H(H, variable_weight=variable_weight)
    if variable_weight:
        DV2_H, _, invDE_HT_DV2 = G 
        np.save(f"data/google+/DVH.npy", DV2_H)
        np.save(f"data/google+/invDE_HT_DVH.npy", invDE_HT_DV2)
    else:
        np.save(f"data/google+/G.npy", G)
    print("SAVED PRECOMPUTED MATRICES to npy")
    
    convert_matrices_to_mm("google+", variable_weight=variable_weight)
    print("CONVERTED all MATRICES to mtx")     
    
    
# set label for isolated nodes
def generate_gplus_labels(filename="google+", train_split=0.8):
    labels = np.load(f"data/{filename}/labels.npy")
    H = np.load(f"data/{filename}/features_coo.npy", allow_pickle=True).item()
    N = H.shape[0]
    labels = dict(labels)
    
    for i in range(N):
        if labels.get(i) is None:
            labels[i] = 468 # set label for isolated nodes - hardcoded for now

    labels = np.array(list(labels.items()))
    np.random.shuffle(labels)
    mmwrite(f"data/{filename}/labels.mtx", labels)

    # n = int(train_split*len(labels))
    # train_labels = labels[:n]
    # test_labels = labels[n:]
    
    # mmwrite(f"data/{filename}/train_labels.mtx", train_labels)
    # mmwrite(f"data/{filename}/test_labels.mtx", test_labels)

                
if __name__ == "__main__":
    # generate_gplus()
    print(f"using {NUM_CPUS} cores")
    generate_gplus_matrices(reset=False, variable_weight=False)
    # generate_gplus_labels()