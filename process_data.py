import torch
import numpy as np
from pathlib import Path

data_path = Path("/media/disk4/wtyao/baselines/cbet/cbet_datasets/relay_kitchen")

my_data = Path("/media/disk4/wtyao/hyperpolicy/datasets/core/threading_d1/992")

observations = torch.from_numpy(np.load(data_path / "observations_seq.npy"))
actions = torch.from_numpy(np.load(data_path / "actions_seq.npy"))
masks = torch.from_numpy(np.load(data_path / "existence_mask.npy"))
goals = torch.load(data_path / "onehot_goals.pth")

my_action = torch.from_numpy(np.load(my_data / "action.npy"))
my_state = torch.from_numpy(np.load(my_data / "state.npy"))

pass