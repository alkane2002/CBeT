# import sys
# sys.path.insert(-1, "/media/disk4/wtyao/baselines/cbet/play-to-policy")

from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import models.latent_generators.latent_generator as latent_generator
from torch.distributions import Categorical

import models.libraries.mingpt.model as mingpt_model
import models.libraries.mingpt.trainer as mingpt_trainer
from models.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple
from utils import batch_indexing

from models.resnet import resnet18
from typing import Union, Dict

class MinGPT(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        n_embd_is_per_head: bool = False,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 0.0,
        goal_conditional: Optional[str] = None,
        goal_seq_len: Optional[int] = None,
        goal_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_dim
        self.n_layer = n_layer
        self.n_head = n_head
        if n_embd_is_per_head:
            self.n_embd = n_head * n_embd
        else:
            self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets
        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma
        self.goal_conditional = goal_conditional
        self.goal_seq_len = goal_seq_len
        self.goal_dim = goal_dim

        if goal_conditional is not None:
            assert goal_conditional in ["prepend", "concat"]
            if goal_conditional == "prepend":
                assert (
                    goal_seq_len is not None
                ), "goal_seq_len must be specified for prepend goal conditional"
            elif goal_conditional == "concat":
                assert (
                    goal_dim is not None
                ), "goal_dim must be specified for concat goal conditional"
        for k, v in kwargs.items():
            setattr(self, k, v)

        effective_vocab_size = self.vocab_size
        if self.predict_offsets:
            effective_vocab_size += self.vocab_size * self.action_dim


        effective_block_size = self.block_size
        if self.goal_conditional == "prepend":
            effective_block_size += self.goal_seq_len

        effective_input_size = self.input_size
        if self.goal_conditional == "concat":
            effective_input_size += self.goal_dim

        gpt_config = mingpt_model.GPTConfig(
            input_size=effective_input_size,
            vocab_size=effective_vocab_size,
            block_size=effective_block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.model = mingpt_model.GPT(gpt_config)
        self.resnet = resnet18()
        x = 1

    def _predict(
        self,
        obs_rep: Union[torch.Tensor, Dict[str, torch.Tensor]],
        goal: Optional[torch.Tensor] = None,
    ):
        """
        Run a forward pass given observation representations and (optionally) goals.
        Arguments:
            obs_rep: Tensor[N, T, E], observation representations.
            goal: Tensor[N, T_goal, G], goal conditionals.
            N: batch size, T: sequence length, E: observation embedding size, G: goal size.
        Returns:
            A batch of predicted actions.
        """
        # These following codes are for modifying the input
        if not hasattr(self, 'resnet') or self.resnet is None:
            self.resnet = resnet18().to("cuda")
            self.goal_conditional = "concat"

        if isinstance(obs_rep, dict):
            img_emb = self.resnet(obs_rep["imgs"]) # [256,2,64]
            obs_rep = torch.cat([img_emb, obs_rep["obs"]], dim=-1) # [256,2,73]
            goal = self.resnet(goal)

        if self.goal_conditional is not None:
            assert goal is not None, "goal must be specified for goal conditional"
            if self.goal_conditional == "prepend":
                is_valid_goal_shape = (
                    goal.shape[0] == obs_rep.shape[0]  # batch size
                    and goal.shape[-1] == obs_rep.shape[-1]  # obs embedding size
                    and goal.shape[1] == self.goal_seq_len
                )
                assert (
                    is_valid_goal_shape
                ), "prepend goal must have shape [batch_size, goal_seq_len, obs_dim]"
                input = torch.cat(
                    [goal, obs_rep], dim=1
                )  # [N, goal_seq_len + T, obs_dim]
            elif self.goal_conditional == "concat":
                # is_valid_goal_shape = (
                #     goal.shape[0] == obs_rep.shape[0]
                #     and goal.shape[1] == obs_rep.shape[1]
                #     and goal.shape[-1] == self.goal_dim
                # )
                # assert (
                #     is_valid_goal_shape
                # ), "concat goal must have shape [batch_size, T, goal_dim]"
                # input = torch.cat([obs_rep, goal], dim=2)
                goal = goal.unsqueeze(1).expand(-1, obs_rep.shape[1], -1)
                input = torch.cat([obs_rep, goal], dim=2)
        else:
            assert goal is None, "goal must be None for non-goal conditional"
            input = obs_rep
        output, _ = self.model(input) # [256,2,512]
        if self.goal_conditional == "prepend":
            output = output[:, self.goal_seq_len :, :]  # [N, T, *]
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> N T V A",
                V=self.vocab_size,
                A=self.action_dim,
            )
            return logits, offsets # ([256, 2, 64], [256, 2, 64, 7])
        else:
            return output

    def _calc_loss(self, output: torch.Tensor, target_latents: torch.Tensor):
        if self.predict_offsets:
            logits, offsets = output
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )
        flat_logits = logits.reshape(-1, logits.size(-1))
        if is_soft_target:
            criterion = soft_cross_entropy
            flat_target_latents = target_latents.view(-1, target_latents.size(-1))
        else:
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
            flat_target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                flat_target_latents = torch.zeros_like(flat_target_latents)

        class_loss = criterion(flat_logits, flat_target_latents)
        loss_components = {"class": class_loss}
        if self.predict_offsets:
            if is_soft_target:
                selected_offsets = batch_indexing(
                    offsets, target_latents.argmax(dim=-1)
                )
            else:
                selected_offsets = batch_indexing(
                    offsets, target_latents.squeeze(-1)
                )  # [N, T, V, A] index [N T](0..V-1) => [N T A]
            offset_loss = self.offset_loss_scale * F.mse_loss(
                selected_offsets, target_offsets
            )
            loss = class_loss + offset_loss
            loss_components["offset"] = offset_loss
        else:
            loss = class_loss
        loss_components["total"] = loss
        return loss, loss_components

    def get_latent_and_loss(
        self,
        obs_rep: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._predict(obs_rep, goal) # ([64,10,64], [64,10,64,9]) -> ([256, 2, 64], [256, 2, 64, 7])
        loss, loss_components = self._calc_loss(output, target_latents)
        if return_loss_components:
            return output, loss, loss_components
        else:
            return output, loss

    def generate_latents(
        self,
        obs_rep: Union[torch.Tensor, Dict[str, torch.Tensor]],
        seq_masks: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # obs_rep = einops.rearrange(seq_obses, "T N E -> N T E")
        # if goal is not None:
        #     goal = einops.rearrange(goal, "T N G -> N T G")

        output = self._predict(obs_rep, goal)
        if self.predict_offsets:
            logits, offsets = output
        else:
            logits = output
        sampled_bins = Categorical(logits=logits).sample()  # batch x seq
        if self.predict_offsets:
            sampled_offsets = batch_indexing(offsets, sampled_bins)  # N T A
            return sampled_bins.unsqueeze(-1), sampled_offsets
        else:
            return sampled_bins.unsqueeze(-1)

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        trainer_cfg = mingpt_trainer.TrainerConfig(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )
        optimizer = self.model.configure_optimizers(trainer_cfg)
        optimizer.add_param_group({"params": self.resnet.parameters()})
        return optimizer


if __name__=="__main__":
    device = "cuda"
    obs_propri = torch.randn(256, 2, 9)
    imgs = torch.randn(256, 2, 3, 128, 128)
    goal = torch.randn(256, 3, 128, 128)
    actions = torch.randn(256, 5, 7)
    obs_rep = {"imgs": imgs.to(device), "obs": obs_propri.to(device)}
    goal = goal.to(device)

    model = MinGPT(discrete_input = False,
            input_dim = 73, # maybe change
            n_layer = 6,
            n_head = 6,
            n_embd = 120,
            n_embd_is_per_head = False,
            block_size = 2, # Length of history/context
            predict_offsets = True,
            offset_loss_scale = 1000.0,
            focal_loss_gamma = 2.0,
            action_dim = 7,
            goal_conditional = "concat", # prepend
            goal_dim = 64,
            goal_seq_len = 2,).to(device)

    x = 1
    
    output = model._predict(obs_rep, goal) # ([256, 2, 64], [256, 2, 64, 7])
    x = 1
    # input = torch.rand(256,2,137)
    # output, _ = self.model(input)
