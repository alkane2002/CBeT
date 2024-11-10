import torch
import numpy as np

observations1 = np.load("/media/disk4/wtyao/baselines/cbet/cbet_datasets/relay_kitchen/observations_seq.npy")
observations2 = torch.from_numpy(np.load("/media/disk4/wtyao/baselines/cbet/cbet_datasets/relay_kitchen/observations_seq.npy"))

observations3 = torch.from_numpy(np.load("/media/disk4/wtyao/baselines/cbet/cbet_datasets/relay_kitchen/observations_seq.npy"))
x = 1