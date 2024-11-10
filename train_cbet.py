import os
import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

import utils
# import wandb
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import re
import numpy as np
from collections import defaultdict, OrderedDict
import os.path as osp
import copy
import pickle
import h5py


normalize_threshold = 5e-2

def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices

def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = data.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
            # normalize to [-1, 1]
            ndata[:, i] = ndata[:, i] * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    data = ndata.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (ndata[:, i] + 1) / 2
            data[:, i] = (
                ndata[:, i] * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
            )
    return data

def check_dir_empty(path):
  """Return True if a directory is empty."""
  with os.scandir(path) as it:
    return not any(it)

def get_subdirs(
    d,
    nonempty = False,
    basename = False,
    sort_lexicographical = False,
    sort_numerical = False,
):
    """Return a list of subdirectories in a given directory.

    Args:
    d: The path to the directory.
    nonempty: Only return non-empty subdirs.
    basename: Only return the tail of the subdir paths.
    sort_lexicographical: Lexicographical sort.
    sort_numerical: Numerical sort.

    Returns:
    The list of subdirectories.
    """
    # Note: `iterdir()` does not yield special entries '.' and '..'.
    subdirs = [f for f in Path(d).iterdir() if f.is_dir()]
    if nonempty:
    # Eliminate empty directories.
        subdirs = [f for f in subdirs if not check_dir_empty(f)]
    if sort_lexicographical:
        subdirs = sorted(subdirs, key=lambda x: x.stem)
    if sort_numerical:
        subdirs = sorted(subdirs, key=lambda x: int(x.stem))
    if basename:
    # Only return the directory stem.
        subdirs = [f.stem for f in subdirs]
    return [str(f) for f in subdirs]

class RobomimicBCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        pred_horizon=4,
        obs_horizon=2,
        action_horizon=4,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        proto_horizon=None,
        mask=None,
        obs_image_based=True,
        unnormal_list=[],
        seed=0,
        use_wrist=False,
        use_alldemo = True,
        train_ratio=0.95,
        train=True,
    ):
        self.patch_size = patch_size
        self.train = train
        self.train_ratio = train_ratio
        self.image_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(self.patch_size),transforms.ToTensor()])
        
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        if mask is not None and os.path.exists(mask) and not use_alldemo:
            self.mask = np.load(mask).tolist()
        else:
            self.mask = None

        self.use_wrist = use_wrist
        self.use_alldemo = use_alldemo
        self.seed = seed
        self.set_seed(self.seed)
        self.obs_image_based = obs_image_based
        self.unnormal_list = unnormal_list

        self.data_dirs = data_dirs
        self._train_or_valid_mask()

        train_data = defaultdict(list)
        self.load_data(train_data)
        self.task_name = os.path.splitext(os.path.basename(data_dirs[0]))[0]

        episode_ends = []
        for eps_action_data in train_data["actions"]:
            episode_ends.append(len(eps_action_data))
            
        self.episode_length = episode_ends

        for k, v in train_data.items():
            train_data[k] = np.concatenate(v)

        print(f"training ({self.train}) data len {len(train_data['actions'])}")

        self.episode_ends = np.cumsum(episode_ends)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "images" or key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)
                train_data[key] = normalize_data(data, stats[key])
                
        self.indices = indices
        self.stats = stats
        self.normalized_train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        if proto_horizon is None:
            self.proto_horizon = obs_horizon
        else:
            self.proto_horizon = proto_horizon

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset."""
        self._dir_tree = OrderedDict()
        for i, path in enumerate(self.data_dirs):
            vids = get_subdirs(path, nonempty=False, sort_numerical=True)
            if vids:
                vids = np.array(vids)
                vids_length = len(vids)
                
                if self.use_alldemo:
                    bool_mask = [True for x in range(vids_length)]
                    vids = vids[bool_mask]
                else:
                    bool_mask = [False for x in range(vids_length)]
                    for id in self.mask:
                        bool_mask[id] = True
                    assert self.mask is not None
                    vids = vids[bool_mask]
                
                num_train = int(vids_length * self.train_ratio)
                sequences = copy.deepcopy(vids)
                random.shuffle(sequences)
                if self.train:
                    vids = sequences[:num_train]
                else:
                    vids = sequences[num_train:]
                vids = sorted(vids, key=lambda x: int(re.search(r'\d+$', x).group()))
                    
                self._dir_tree[path] = vids

    def _train_or_valid_mask(self):
        """Build a dict of indices for iterating over the dataset."""
        traj_nums = 1000
        sequences = list(range(1000))
                
        num_train = int(traj_nums * self.train_ratio)
        random.shuffle(sequences)
        if self.train:
            vids = sequences[:num_train]
        else:
            vids = sequences[num_train:]
        vids = sorted(vids)
        
        def read_hdf5_file(file_path):
            with h5py.File(file_path, 'r') as f:
                data = {}
                for i in vids:
                    demo_key = f'demo_{i}'
                    if demo_key not in data:
                        data[demo_key] = {'obs': {}, 'actions': {}}
                    data[demo_key]['obs']['robot0_eef_pos']=f[f'data/{demo_key}/obs/robot0_eef_pos'][:]
                    data[demo_key]['obs']['robot0_eef_quat']=f[f'data/{demo_key}/obs/robot0_eef_quat'][:]
                    data[demo_key]['obs']['robot0_gripper_qpos']=f[f'data/{demo_key}/obs/robot0_gripper_qpos'][:]
                    data[demo_key]['obs']['agentview_image']=f[f'data/{demo_key}/obs/agentview_image'][:]
                    data[demo_key]['actions']=f[f'data/{demo_key}/actions'][:]
            return data

        self.all_data = read_hdf5_file(self.data_dirs[0])
        self.demo_key_masks = vids
   
    def load_action_and_to_tensor(self, vid):
        action_path = os.path.join(vid, "action.npy")
        action_data = np.load(action_path)
        action_data = np.array(action_data, dtype=np.float32)
        return action_data

    def load_state_and_to_tensor(self, vid):
        state_path = os.path.join(vid, "state.npy")
        state_data = np.load(state_path)
        state_data = np.array(state_data, dtype=np.float32)
        return state_data

    def load_data(self, train_data):
        # HACK. Fix later
        data = self.all_data
        for i in range(0, len(self.demo_key_masks)):
            data_obs = data[f"demo_{self.demo_key_masks[i]}"]["obs"]
            obs = np.concatenate((data_obs['robot0_eef_pos'], data_obs['robot0_eef_quat'], data_obs['robot0_gripper_qpos']), axis=1)[:]
            acs = data[f"demo_{self.demo_key_masks[i]}"]["actions"][:]
            train_data["obs"].append(np.array(obs, dtype=np.float32))
            train_data["actions"].append(np.array(acs, dtype=np.float32))
    
    def load_and_resize_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        return image

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx,) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        
        # IMPORTANT: By default the length of the action prediction will be greater than the length of the obs
        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :9]
        nsample["task_name"] = self.task_name
        
        if self.obs_image_based:
            for i, episode_end_idx in enumerate(self.episode_ends):
                if buffer_start_idx >= episode_end_idx:
                    continue
                
                imgs_dir = self.demo_key_masks[i]
                
                start_idx = buffer_start_idx if i == 0 else buffer_start_idx - self.episode_ends[i-1]
                if self.obs_horizon <= buffer_end_idx - buffer_start_idx:
                    end_idx = start_idx + self.obs_horizon
                else:
                    end_idx = start_idx + (buffer_end_idx - buffer_start_idx)
                    
                if sample_start_idx==0 and sample_end_idx==self.pred_horizon:
                    start_idx = buffer_start_idx if i == 0 else buffer_start_idx - self.episode_ends[i-1]
                    end_idx = start_idx + self.obs_horizon
                elif sample_end_idx != self.pred_horizon:
                    start_idx = buffer_start_idx if i == 0 else buffer_start_idx - self.episode_ends[i-1]
                    end_idx = start_idx + min((buffer_end_idx - buffer_start_idx),self.obs_horizon)
                else:
                    start_idx = buffer_start_idx if i == 0 else buffer_start_idx - self.episode_ends[i-1]
                    end_idx = start_idx + (buffer_end_idx - buffer_start_idx)-(self.pred_horizon-self.obs_horizon)
                    
                images = []
                if self.use_wrist:
                    wrist_images = []
                data = self.all_data
                for img_idx in range(start_idx, end_idx):
                    images.append(self.image_transform(data[f"demo_{imgs_dir}"]["obs"]['agentview_image'][img_idx]))
                    if self.use_wrist:
                        wrist_images.append(self.image_transform(data[f"demo_{imgs_dir}"]["obs"]['robot0_eye_in_hand_image'][img_idx]))

                # rand_goal_idx = random.randint(end_idx, self.episode_length[i]-1)
                # rand_start_idx = random.randint(0, start_idx)

                # nsample["goal_image"] = self.load_and_resize_image(os.path.join(imgs_dir, f"{self.episode_length[i]-1}.png"))
                if (end_idx-1)>(self.episode_length[i]-1):
                    x = 1

                # The following two lines aim to randomly sample the goal images and the start image.
                # nsample["goal_image"] = self.load_and_resize_image(os.path.join(imgs_dir, f"{random.randint((end_idx-1), self.episode_length[i]-1)}.png"))
                # nsample["start_image"] = self.load_and_resize_image(os.path.join(imgs_dir, f"{random.randint(0, start_idx)}.png"))

                # ========================= different goal settings =========================
                # # The following two lines aim to specificly sample the goal images and the start image.
                # goal_idx = self.episode_length[i]-1
                # nsample["goal_image"] = self.load_and_resize_image(os.path.join(imgs_dir, f"{goal_idx}.png"))
                
                # ---------------------mimicgen dataset---------------------
                # random subgoal 
                if (start_idx + self.pred_horizon +1 >= self.episode_length[i]-1):
                    goal_idx = self.episode_length[i]-1
                else:
                    goal_idx = random.randint(start_idx + self.pred_horizon + 1, self.episode_length[i]-1)

                nsample["goal_image"] = self.image_transform(data[f"demo_{imgs_dir}"]["obs"]['agentview_image'][goal_idx])
                
                nsample["images"] = torch.stack(images)
                if self.use_wrist:
                    nsample["wrist_images"] = torch.stack(wrist_images)
                    nsample["goal_wrist_image"] = self.image_transform(data[f"demo_{imgs_dir}"]["obs"]['robot0_eye_in_hand_image'][goal_idx])

                nsample["start_image"] = images[-1] # current image
                next_idx = min(end_idx, self.episode_length[i]-1)
                nsample["next_image"] = self.image_transform(data[f"demo_{imgs_dir}"]["obs"]['agentview_image'][next_idx])
                
                nsample["next_img_idx"] = end_idx/(self.episode_length[i]-1)
                nsample["curr_img_idx"] = (end_idx-1)/(self.episode_length[i]-1)
                
                # Extend images array if not enough images were loaded
                if len(images) < self.obs_horizon:
                    if sample_start_idx > 0:
                        additional_slices = torch.repeat_interleave(nsample["images"][0:1], self.obs_horizon - len(images), axis=0)
                        nsample["images"] = torch.cat([additional_slices, nsample["images"]], axis=0)
                        if self.use_wrist:
                            additional_slices = torch.repeat_interleave(nsample["wrist_images"][0:1], self.obs_horizon - len(images), axis=0)
                            nsample["wrist_images"] = torch.cat([additional_slices, nsample["wrist_images"]], axis=0)
                    if sample_end_idx < self.pred_horizon:
                        additional_slices = torch.repeat_interleave(nsample["images"][-1:], self.obs_horizon - len(images), axis=0)
                        nsample["images"] = torch.cat([nsample["images"], additional_slices], axis=0)
                        if self.use_wrist:
                            additional_slices = torch.repeat_interleave(nsample["wrist_images"][-1:], self.obs_horizon - len(images), axis=0)
                            nsample["wrist_images"] = torch.cat([nsample["wrist_images"], additional_slices], axis=0)
                break
        return nsample

class Workspace:
    def __init__(self, cfg):
        # self.work_dir = Path.cwd()

        task_name = "coffee_d1"

        self.work_dir = f"/media/disk4/wtyao/hyperpolicy/EXP/c_bet/{task_name}/{os.path.basename(Path.cwd()).split('_')[0]}"
        
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        if HydraConfig.get().mode == RunMode.MULTIRUN:
            self.job_num = HydraConfig.get().job.num
        else:
            self.job_num = 0

        cuda_idx = 5
        self.device = torch.device(f"cuda:{cuda_idx}")


        utils.set_seed_everywhere(cfg.seed)
        # self.dataset = hydra.utils.call(
        #     cfg.env.dataset_fn,
        #     train_fraction=cfg.train_fraction,
        #     random_seed=cfg.seed,
        #     device=self.device,
        # )
        # self.train_set, self.test_set = self.dataset
        # self.train_set = RobomimicBCDataset("/media/disk4/wtyao/mimicgen/mimicgen/mimicgen_videos/core/coffee_d0", train=True)
        # self.test_set = RobomimicBCDataset("/media/disk4/wtyao/mimicgen/mimicgen/mimicgen_videos/core/coffee_d0", train=False)
        
        print("Loading dataset......")
        self.train_set = RobomimicBCDataset(f"/media/disk4/wtyao/mimicgen/mimicgen/datasets/core/{task_name}.hdf5", train=True)
        self.test_set = RobomimicBCDataset(f"/media/disk4/wtyao/mimicgen/mimicgen/datasets/core/{task_name}.hdf5", train=False)
        
        stats_dict = {}
        stats_dict[task_name] = self.train_set.stats
        if not self.work_dir is None:
            os.makedirs(self.work_dir, exist_ok=True)  # exist_ok=True
            with open(os.path.join(self.work_dir, "stats.pickle"), "wb") as f:
                pickle.dump(stats_dict, f)

        self._setup_loaders()

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        self.log_components = OrderedDict()
        self.epoch = self.prior_epoch = 0

        self.save_training_latents = False
        self._training_latents = []

        # self.wandb_run = wandb.init(
        #     dir=str(self.work_dir),
        #     project=cfg.project,
        #     reinit=False,
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     settings=dict(start_method="thread"),
        # )
        # logging.info("wandb run url: %s", self.wandb_run.get_url())
        # wandb.config.update(
        #     {
        #         "save_path": self.work_dir,
        #     }
        # )

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)
            if self.cfg.data_parallel:
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)
            if self.cfg.data_parallel:
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            total_params, table = utils.count_parameters(self.state_prior)
            logging.info(table)
            logging.info(
                "Total number of parameters in state prior: {}".format(total_params)
            )
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.latent_collection_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def train_prior(self):
        self.state_prior.train()
        to_freeze = [self.obs_encoding_net]
        if not self.cfg.train_encoder_with_prior:
            to_freeze.append(self.action_ae)
        with utils.eval_mode(*to_freeze):
            pbar = tqdm.tqdm(
                self.train_loader, desc=f"Training prior epoch {self.prior_epoch}"
            )
            for data in pbar:
                if self.cfg.goal_conditional is not None:
                    # observations, actions, mask, goal = data # ([64, 10, 60], [64, 10, 9], [64, 10], [64, 10, 60])
                    obs_propri = data["obs"] # obs_propri = torch.randn(256, 2, 9)
                    imgs = data["images"] # imgs = torch.randn(256, 2, 3, 128, 128)
                    goal = data["goal_image"] # goal = torch.randn(256, 3, 128, 128)
                    actions = data["actions"] # actions = torch.randn(256, 5, 7)
                    actions = actions[:,1:3,:] # The second dim of actions should be equal to that of obs
                    obs = {"imgs": imgs.to(self.device), "obs": obs_propri.to(self.device)}
    
                    goal = goal.to(self.device)
                    if self.cfg.goal_conditional == "future":
                        enc_goal = self.obs_encoding_net(goal)
                    else:
                        enc_goal = goal
                else:
                    observations, actions, mask = data
                    goal = None
                    enc_goal = None
                act = actions.to(self.device)
                enc_obs = self.obs_encoding_net(obs)

                if self.state_prior_optimizer is not None:
                    self.state_prior_optimizer.zero_grad(set_to_none=True)
                if self.cfg.train_encoder_with_prior:
                    self.action_ae.optimizer.zero_grad(set_to_none=True)

                ae_loss = 0
                ae_loss_components = {}
                if self.cfg.train_encoder_with_prior:
                    (
                        latent,
                        ae_loss,
                        ae_loss_components,
                    ) = self.action_ae.calculate_encodings_and_loss(
                        act, enc_obs, return_loss_components=True
                    )
                else:
                    latent = self.action_ae.encode_into_latent(act, enc_obs) # it's a tuple of tensors
                if type(latent) is tuple:
                    latent = (x.detach() for x in latent)
                else:
                    latent = latent.detach()
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    goal=enc_goal,
                    return_loss_components=True,
                )
                total_loss = loss + ae_loss
                if self.state_prior_optimizer is not None:
                    total_loss.backward()
                if self.cfg.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.state_prior.parameters(), self.cfg.grad_norm_clip
                    )
                    if self.cfg.train_encoder_with_prior:
                        torch.nn.utils.clip_grad_norm_(
                            self.action_ae.parameters(), self.cfg.grad_norm_clip
                        )
                if self.state_prior_optimizer is not None:
                    self.state_prior_optimizer.step()
                if self.cfg.train_encoder_with_prior:
                    self.action_ae.optimizer.step()
                    self.log_append("ae_train", len(observations), ae_loss_components)
                self.log_append("prior_train", len(actions), loss_components)

        if hasattr(self.state_prior, "finish_training"):
            self.state_prior.finish_training()  # type: ignore

    def eval_prior(self):
        with utils.eval_mode(
            self.obs_encoding_net, self.action_ae, self.state_prior, no_grad=True
        ):
            self.eval_loss = 0
            for data in self.test_loader:
                if self.cfg.goal_conditional is not None:
                    # observations, actions, mask, goal = data
                    obs_propri = data["obs"] # obs_propri = torch.randn(256, 2, 9)
                    imgs = data["images"] # imgs = torch.randn(256, 2, 3, 128, 128)
                    goal = data["goal_image"] # goal = torch.randn(256, 3, 128, 128)
                    actions = data["actions"] # actions = torch.randn(256, 5, 7)
                    actions = actions[:,1:3,:] # The second dim of actions should be equal to that of obs
                    obs = {"imgs": imgs.to(self.device), "obs": obs_propri.to(self.device)}

                    goal = goal.to(self.device)
                    if self.cfg.goal_conditional == "future":
                        enc_goal = self.obs_encoding_net(goal)
                else:
                    observations, actions, mask = data
                    goal = None
                    enc_goal = None
                act = actions.to(self.device)
                enc_obs = self.obs_encoding_net(obs)

                if hasattr(self.action_ae, "calculate_encodings_and_loss"):
                    (
                        latent,
                        ae_loss,
                        ae_loss_components,
                    ) = self.action_ae.calculate_encodings_and_loss(
                        act, enc_obs, return_loss_components=True
                    )
                else:
                    latent = self.action_ae.encode_into_latent(act, enc_obs)
                    ae_loss, ae_loss_components = 0, {}
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    goal=goal,
                    return_loss_components=True,
                )
                self.eval_loss += loss * len(actions)
                self.log_append("ae_eval", len(actions), ae_loss_components)
                self.log_append("prior_eval", len(actions), loss_components)
            self.eval_loss /= len(self.test_set)

    def run(self):
        snapshot = self.snapshot
        if snapshot.exists():
            print(f"Resuming: {snapshot}")
            self.load_snapshot()

        if self.cfg.lazy_init_models:
            self._init_obs_encoding_net()
            self._init_action_ae()
            
        self.action_ae.fit_model(
            self.train_loader,
            self.test_loader,
            self.obs_encoding_net,
        )
        if self.cfg.save_latents:
            self.save_latents()

        # Train the action prior model.
        if self.cfg.lazy_init_models:
            self._init_state_prior()
        self.state_prior_iterator = tqdm.trange(
            self.prior_epoch, self.cfg.num_prior_epochs
        )
        self.state_prior_iterator.set_description("Training prior: ")
        # Reset the log.
        self.log_components = OrderedDict()
        for epoch in self.state_prior_iterator:
            self.prior_epoch = epoch
            self.train_prior()
            if ((self.prior_epoch + 1) % self.cfg.eval_prior_every) == 0:
                self.eval_prior()
            self.flush_log(epoch=epoch + self.epoch, iterator=self.state_prior_iterator)
            self.prior_epoch += 1
            if ((self.prior_epoch + 1) % self.cfg.save_prior_every) == 0:
                self.save_snapshot()

        # # expose DataParallel module class name for wandb tags
        # tag_func = (
        #     lambda m: m.module.__class__.__name__
        #     if self.cfg.data_parallel
        #     else m.__class__.__name__
        # )
        # tags = tuple(
        #     map(tag_func, [self.obs_encoding_net, self.action_ae, self.state_prior])
        # )
        # self.wandb_run.tags += tags
        return float(self.eval_loss)

    @property
    def snapshot(self):
        return Path(self.work_dir) / "snapshot.pt"

    def save_snapshot(self):
        self._keys_to_save = [
            "action_ae",
            "obs_encoding_net",
            "epoch",
            "prior_epoch",
            "state_prior",
        ]
        payload = {k: self.__dict__[k] for k in self._keys_to_save}
        with self.snapshot.open("wb") as f:
            torch.save(payload, f)

    def save_latents(self):
        total_mse_loss = 0
        with utils.eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, actions, mask in self.latent_collection_loader:
                obs, act = observations.to(self.device), actions.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    self._training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    self._training_latents.append(latent.detach())
        self._training_latents_tensor = torch.cat(self._training_latents, dim=0)
        logging.info(f"Total MSE reconstruction loss: {total_mse_loss}")
        logging.info(
            f"Average MSE reconstruction loss: {total_mse_loss / len(self._training_latents_tensor)}"
        )
        torch.save(self._training_latents_tensor, self.work_dir / "latents.pt")

    def load_snapshot(self):
        with self.snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        not_in_payload = set(self._keys_to_save) - set(payload.keys())
        if len(not_in_payload):
            logging.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        iterator.set_postfix_str(postfix)
        # wandb.log(log_components, step=epoch)
        logging.info(f"[{self.job_num}] Epoch {epoch}: {log_components}")
        self.log_components = OrderedDict()


@hydra.main(version_base="1.2", config_path="configs", config_name="train_kitchen_future_cond")
def main(cfg):
    workspace = Workspace(cfg)
    eval_loss = workspace.run()
    return eval_loss


if __name__ == "__main__":
    main()
    
