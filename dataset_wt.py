import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import re
import numpy as np
from tqdm import tqdm
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

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
    
class DemonstrationDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, train_ratio=0.95):
        self.root_dir = root_dir
        self.transform = transform if transform else ToTensor()
        self.train = train

        sequences = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        random.shuffle(sequences)
        
        num_train = int(len(sequences) * train_ratio)
        if self.train:
            self.selected_sequences = sequences[:num_train]
        else:
            self.selected_sequences = sequences[num_train:]

        self.images = []
        for seq in self.selected_sequences:
            self.images.extend([os.path.join(seq, img) for img in os.listdir(seq) if re.match(r'^\d+\.png$', img)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0
    
class DemonVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        print(data_path)
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = DemonstrationDataset(self.data_dir, train=True, transform=train_transforms)
        self.val_dataset = DemonstrationDataset(self.data_dir, train=False, transform=val_transforms)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

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

class RobomimicBCDataset_Lightning(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        print(data_path)
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, task_names: List[str], stats_dir: Optional[str] = None, stage: Optional[str] = None) -> None:
        train_dataset_list = []
        val_dataset_list = []
        stats_dict = {}
        for task_name in task_names:
            new_data_path = os.path.join(self.data_dir, f"{task_name}.hdf5")
            new_data_path = new_data_path if os.path.exists(new_data_path) else None
            
            train_dataset = RobomimicBCDataset(new_data_path, train=True, patch_size=self.patch_size, **self.kwargs)
            val_dataset = RobomimicBCDataset(new_data_path, train=False, patch_size=self.patch_size, **self.kwargs)
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
            stats_dict[task_name] = train_dataset.stats
            
        self.train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
        self.val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
        if not stats_dir is None:
            with open(os.path.join(stats_dir, "stats.pickle"), "wb") as f:
                pickle.dump(stats_dict, f)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

if __name__=="__main__":
    # data_dirs = '/media/disk4/wtyao/hyperpolicy/coffee_d0'
    # dataset = RobomimicBCDataset(data_dirs=data_dirs)
    # tmp = dataset[101]
    
    dataset = RobomimicBCDataset("/media/disk4/wtyao/mimicgen/mimicgen/datasets/core/coffee_d0.hdf5", train=True)
    tmp = dataset[101]

    # obs: (2, 9)
    # actions: (4, 7)   timestep = 4, action dimention = 7
    # goal_image: torch.Size([3, 256, 256])
    # start_image: torch.Size([3, 256, 256])
    # next_image: torch.Size([3, 256, 256])
    # next_img_idx: ()
    # curr_img_idx: ()
    # images: torch.Size([2, 3, 256, 256])

    for key, value in tmp.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
    
    x = 1
    
    
    