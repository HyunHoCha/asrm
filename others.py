import utils
from utils import *
import pickle

# python3 others.py

# ----------------------------------------------------------------------------------------------------
# make training data
N = 2
C = 3
num_data = 10000
data = []
for i in range(num_data):
    sigma_list, probs, dist_list, PGM_prob, purities = POVM_NC_innerDistance_purity(N, C)
    data.append((sigma_list, probs, dist_list, PGM_prob, purities))
    if i % 1000 == 0:
        print("making...")
with open('train/dim' + str(N) + 'cls' + str(C) + 'InnerDistancePurity.pkl', 'wb') as f:
    pickle.dump(data, f)
# ----------------------------------------------------------------------------------------------------
# make validation data
N = 2
C = 3
num_data = 1000
data = []
for i in range(num_data):
    sigma_list, probs, dist_list, PGM_prob, purities = POVM_NC_innerDistance_purity(N, C)
    data.append((sigma_list, probs, dist_list, PGM_prob, purities))
    if i % 100 == 0:
        print("making...")
with open('val/dim' + str(N) + 'cls' + str(C) + 'InnerDistancePurity.pkl', 'wb') as f:
    pickle.dump(data, f)
# ----------------------------------------------------------------------------------------------------
