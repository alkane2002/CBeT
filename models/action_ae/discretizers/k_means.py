import torch
import numpy as np

import tqdm

from typing import Optional, Tuple, Union
from models.action_ae.discretizers.base import AbstractDiscretizer


class KMeansDiscretizer(AbstractDiscretizer):
    """
    Simplified and modified version of KMeans algorithm  from sklearn.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 100,
        device: Union[str, torch.device] = "cpu",
        predict_offsets: bool = False,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.device = device
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets

    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        assert (
            self.action_dim == input_actions.shape[-1]
        ), f"Input action dimension {self.action_dim} does not match fitted model {input_actions.shape[-1]}"

        flattened_actions = input_actions.view(-1, self.action_dim)
        cluster_centers = KMeansDiscretizer._kmeans(
            flattened_actions, ncluster=self.n_bins, device=self.device,
        )
        self.bin_centers = cluster_centers.to(self.device)

    @property
    def suggested_actions(self) -> torch.Tensor:
        return self.bin_centers

    # @classmethod
    # def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50, device="cpu"):
    #     """
    #     Simple k-means clustering algorithm adapted from Karpathy's minGPT libary
    #     https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
    #     """
    #     N, D = x.size()
    #     c = x[torch.randperm(N)[:ncluster]]  # init clusters at random

    #     pbar = tqdm.trange(niter)
    #     pbar.set_description("K-means clustering")
    #     for i in pbar:
    #         # assign all pixels to the closest codebook element
    #         a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
    #         # move each codebook element to be the mean of the pixels that assigned to it
    #         c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
    #         # re-assign any poorly positioned codebook elements
    #         nanix = torch.any(torch.isnan(c), dim=1)
    #         ndead = nanix.sum().item()
    #         if ndead:
    #             tqdm.tqdm.write(
    #                 "done step %d/%d, re-initialized %d dead clusters"
    #                 % (i + 1, niter, ndead)
    #             )
    #         c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
    #     return c
    
    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50, device="cpu"):
        """
        GPU-accelerated k-means clustering
        """
        N, D = x.size()
        x = x.to(device) 
        idx = torch.randperm(N, device=device)[:ncluster]
        c = x[idx].clone()

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means clustering")

        distances = torch.empty(N, ncluster, device=device)
        
        for i in pbar:
            batch_size = 1024 
            for b in range(0, N, batch_size):
                end = min(b + batch_size, N)
                distances[b:end] = ((x[b:end, None, :] - c[None, :, :]) ** 2).sum(-1)
            
            a = distances.argmin(1)
            
            for k in range(ncluster):
                mask = (a == k)
                if mask.any():
                    c[k] = x[mask].mean(0)
            
            empty_clusters = torch.where(torch.isnan(c).any(dim=1))[0]
            n_empty = len(empty_clusters)
            
            if n_empty > 0:
                tqdm.tqdm.write(
                    f"Step {i+1}/{niter}, re-initialized {n_empty} empty clusters"
                )
                new_centers_idx = torch.randperm(N, device=device)[:n_empty]
                c[empty_clusters] = x[new_centers_idx]
        return c

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given the input action, discretize it using the k-Means clustering algorithm.

        Inputs:
        input_action (shape: ... x action_dim): The input action to discretize. This can be in a batch,
        and is generally assumed that the last dimnesion is the action dimension.

        Outputs:
        discretized_action (shape: ... x num_tokens): The discretized action.
        If self.predict_offsets is True, then the offsets are also returned.
        """
        assert (
            input_action.shape[-1] == self.action_dim
        ), "Input action dimension does not match fitted model"

        # flatten the input action
        flattened_actions = input_action.view(-1, self.action_dim) #[512,7]

        # get the closest cluster center
        closest_cluster_center = torch.argmin(
            torch.sum(
                (flattened_actions[:, None, :] - self.bin_centers[None, :, :]) ** 2,
                dim=2,
            ),
            dim=1,
        )
        # Reshape to the original shape
        discretized_action = closest_cluster_center.view(input_action.shape[:-1] + (1,)) # [256,2,1]

        if self.predict_offsets:
            # decode from latent and get the difference
            reconstructed_action = self.decode_actions(discretized_action)
            offsets = input_action - reconstructed_action
            return (discretized_action, offsets) # [256,2,1]int, [256,2,7]float
        else:
            # return the one-hot vector
            return discretized_action

    def decode_actions(
        self,
        latent_action_batch: torch.Tensor,
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given the latent action, reconstruct the original action.

        Inputs:
        latent_action (shape: ... x 1): The latent action to reconstruct. This can be in a batch,
        and is generally assumed that the last dimension is the action dimension. If the latent_action_batch
        is a tuple, then it is assumed to be (discretized_action, offsets).

        Outputs:
        reconstructed_action (shape: ... x action_dim): The reconstructed action.
        """
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch
        # get the closest cluster center
        closest_cluster_center = self.bin_centers[latent_action_batch]
        # Reshape to the original shape
        reconstructed_action = closest_cluster_center.view(
            latent_action_batch.shape[:-1] + (self.action_dim,)
        )
        if offsets is not None:
            reconstructed_action += offsets
        return reconstructed_action

    @property
    def discretized_space(self) -> int:
        return self.n_bins

    @property
    def latent_dim(self) -> int:
        return 1

    @property
    def num_latents(self) -> int:
        return self.n_bins
