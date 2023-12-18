import numpy as np
from numpy import savetxt
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, rand
import os
import math

if __name__ == "__main__":
  fts_shape = (12311, 6144)
  dvh_shape = (12311, 24622)
  invDE_HT_DVH_shape = (24622, 12311)
  g_shape = (12311, 12311)
  fts_mean = 0.5580087304115295
  fts_std = 0.8535025119781494
  dvh_mean = 0.00016285375075872916
  dvh_std = 0.005937955496888596
  invDE_HT_DVH_mean = 1.9082127289517364e-05
  invDE_HT_DVH_std = 0.0006997982127210402
  g_mean = 7.718491835159984e-05
  g_std = 0.0015633388801232205
  dvh_nnzs = 246220
  invDE_HT_DVH_nnzs = 246220
  g_nnzs = 863793
  
  print("Generating random matrices")
  for scale in [8, 16]:
    if not os.path.exists(f"/cluster/scratch/siwachte/data/random/scale_{scale}"):
      os.makedirs(f"/cluster/scratch/siwachte/data/random/scale_{scale}")
      
    new_dvh = rand(int(math.sqrt(scale) * dvh_shape[0]), int(math.sqrt(scale) * dvh_shape[1]), density=dvh_nnzs/(dvh_shape[0]*dvh_shape[1]), format='coo')
    with open(f"/cluster/scratch/siwachte/data/random/scale_{scale}/DVH.csv", "w") as f:
      f.write("row,col,val\n")
      for (i,j, v) in zip(new_dvh.row, new_dvh.col, new_dvh.data):
        f.write(f"{i},{j},{v}\n")
      
    mmwrite(f"/cluster/scratch/siwachte/data/random/scale_{scale}/DVH.mtx", coo_matrix(new_dvh))
    print(f"Done with DVH at scale {scale}")
    
    new_invDE_HT_DVH = rand(int(math.sqrt(scale) * invDE_HT_DVH_shape[0]), int(math.sqrt(scale) * invDE_HT_DVH_shape[1]), density=invDE_HT_DVH_nnzs/(invDE_HT_DVH_shape[0]*invDE_HT_DVH_shape[1]), format='coo')
    indexes = np.nonzero(new_invDE_HT_DVH)
    with open(f"/cluster/scratch/siwachte/data/random/scale_{scale}/invDE_HT_DVH.csv", "w") as f:
        f.write("row,col,val\n")
        for (i,j, v) in zip(new_invDE_HT_DVH.row, new_invDE_HT_DVH.col, new_invDE_HT_DVH.data):
          f.write(f"{i},{j},{v}\n")
    mmwrite(f"/cluster/scratch/siwachte/data/random/scale_{scale}/invDE_HT_DVH.mtx", coo_matrix(new_invDE_HT_DVH))
    print(f"Done with invDE_HT_DVH at scale {scale}")
    
    new_g = rand(int(math.sqrt(scale) * g_shape[0]), int(math.sqrt(scale) * g_shape[1]), density=g_nnzs/(g_shape[0]*g_shape[1]), format='coo')
    indexes = np.nonzero(new_g)
    with open(f"/cluster/scratch/siwachte/data/random/scale_{scale}/G.csv", "w") as f:
        f.write("row,col,val\n")
        for (i,j, v) in zip(new_g.row, new_g.col, new_g.data):
          f.write(f"{i},{j},{v}\n")
    mmwrite(f"/cluster/scratch/siwachte/data/random/scale_{scale}/G.mtx", coo_matrix(new_g))
    print(f"Done with G at scale {scale}")
    
    new_fts = np.random.normal(fts_mean, fts_std, size=(int(math.sqrt(scale) * fts_shape[0]), fts_shape[1]))
    new_fts = np.clip(new_fts, 0, None)
    new_fts = np.round(new_fts, 8)
    df = pd.DataFrame(new_fts)
    df.to_csv(f"/cluster/scratch/siwachte/data/random/scale_{scale}/features.csv", index=False, header=False)
    mmwrite(f"/cluster/scratch/siwachte/data/random/scale_{scale}/features.mtx", new_fts, precision=8)
    print(f"Done with features at scale {scale}")