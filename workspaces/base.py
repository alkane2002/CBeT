import logging
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf
import einops
import gym
from gym.wrappers import RecordVideo
import hydra
import numpy as np
import torch
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
# import wandb

from torchvision import transforms
from utils import data_loader
from torchvision import transforms
import cv2
import collections
import pickle
import random
from PIL import Image
import imageio
from copy import deepcopy
import random

if 'eval' in os.environ.get('SCRIPT_NAME', ''):
    # sys.path.insert(-1, "/media/disk4/wtyao/hyperpolicy/")
    # sys.path.insert(-1, "/home/pzhou/xskill/robosuite-task-zoo")
    # sys.path.insert(-1, "/home/pzhou/xskill/mimicgen_environments")
    import robosuite as suite
    from dataset import normalize_data, unnormalize_data
    from robosuite.controllers import load_controller_config
    from robosuite.utils.input_utils import *
    import mimicgen


def generate_key_from_value_corrected(value):
    words = value.split('_')
    suffix = words[-1]  
    transformed = ''.join(word.capitalize() for word in words[:-1])
    key_name = f"{transformed}_{suffix.capitalize()}"
    return key_name

def convert_images_to_tensors(images_arr, pipeline=None):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    if pipeline is not None:
        images_tensor = pipeline(images_tensor)
    return images_tensor

def configure_environment(cfg, options):
    camera_names = ["agentview"]
    if cfg.use_wrist:
        camera_names.append("robot0_eye_in_hand")
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=84,  # set camera height
        camera_widths=84,  # set camera width
        control_freq=20,
        camera_names=camera_names,
    )
    return env

def process_images(obs, camera_name, eval_cfg, obs_horizon):
    image = obs[camera_name]
    image = image[::-1, :, :]  # Flip the image vertically
    resized_image = cv2.resize(image, eval_cfg.bc_resize)
    return collections.deque([resized_image] * obs_horizon, maxlen=obs_horizon)



### backup codes
# class Workspace:
#     def __init__(self, cfg):
#         self.work_dir = Path.cwd()
#         print("Saving to {}".format(self.work_dir))
#         self.cfg = cfg
#         self.device = torch.device(cfg.device)
#         utils.set_seed_everywhere(cfg.seed)

#         self.env = gym.make(cfg.env.name)
#         if self.env.class_name() == "TimeLimit":
#             # wrap with the correct num steps
#             self.env = self.env.env
#             self.env = gym.wrappers.TimeLimit(self.env, cfg.num_eval_steps)
#         if cfg.record_video:
#             self.env = RecordVideo(
#                 self.env,
#                 video_folder=self.work_dir,
#                 episode_trigger=lambda x: x % 1 == 0,
#             )

#         # Create the model
#         self.action_ae = None
#         self.obs_encoding_net = None
#         self.state_prior = None
#         if not self.cfg.lazy_init_models:
#             self._init_action_ae()
#             self._init_obs_encoding_net()
#             self._init_state_prior()

#         if self.cfg.goal_conditional:
#             self.goal_fn = hydra.utils.instantiate(self.cfg.env.goal_fn, cfg)
#         self.wandb_run = wandb.init(
#             dir=self.work_dir,
#             project=cfg.project,
#             config=OmegaConf.to_container(cfg, resolve=True),
#         )
#         logging.info("wandb run url: %s", self.wandb_run.get_url())
#         self.epoch = 0
#         self.load_snapshot()

#         # Set up rolling window contexts.
#         self.window_size = cfg.window_size
#         self.obs_context = deque(maxlen=self.window_size)
#         self.goal_context = deque(maxlen=self.window_size)

#         if self.cfg.flatten_obs:
#             self.env = gym.wrappers.FlattenObservation(self.env)

#         if self.cfg.plot_interactions:
#             self._setup_plots()

#         if self.cfg.start_from_seen:
#             self._setup_starting_state()

#         self._setup_action_sampler()

#     def _init_action_ae(self):
#         if self.action_ae is None:  # possibly already initialized from snapshot
#             self.action_ae = hydra.utils.instantiate(
#                 self.cfg.action_ae, _recursive_=False
#             ).to(self.device)
#             if self.cfg.data_parallel:
#                 self.action_ae = GeneratorDataParallel(self.action_ae)

#     def _init_obs_encoding_net(self):
#         if self.obs_encoding_net is None:  # possibly already initialized from snapshot
#             self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
#             self.obs_encoding_net = self.obs_encoding_net.to(self.device)
#             if self.cfg.data_parallel:
#                 self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

#     def _init_state_prior(self):
#         if self.state_prior is None:  # possibly already initialized from snapshot
#             self.state_prior = hydra.utils.instantiate(
#                 self.cfg.state_prior,
#                 latent_dim=self.action_ae.latent_dim,
#                 vocab_size=self.action_ae.num_latents,
#             ).to(self.device)
#             if self.cfg.data_parallel:
#                 self.state_prior = LatentGeneratorDataParallel(self.state_prior)
#             self.state_prior_optimizer = self.state_prior.get_optimizer(
#                 learning_rate=self.cfg.lr,
#                 weight_decay=self.cfg.weight_decay,
#                 betas=tuple(self.cfg.betas),
#             )

#     def _setup_plots(self):
#         raise NotImplementedError

#     def _setup_starting_state(self):
#         raise NotImplementedError

#     def _setup_action_sampler(self):
#         def sampler(actions):
#             idx = np.random.randint(len(actions))
#             return actions[idx]

#         self.sampler = sampler

#     def _start_from_known(self):
#         raise NotImplementedError

#     def run_single_episode(self, goal_idx=None):
#         action_history = []
#         obs = self.env.reset()
#         last_obs = obs
#         if self.cfg.start_from_seen:
#             obs = self._start_from_known()
#         if self.cfg.goal_conditional:
#             goal = self.goal_fn(obs, goal_idx, 0)
#         else:
#             goal = None
#         action = self._get_action(obs, goal=goal, sample=True)
#         done = False
#         total_reward = 0
#         action_history.append(action)
#         for i in range(self.cfg.num_eval_steps + 1):
#             if self.cfg.plot_interactions:
#                 self._plot_obs_and_actions(obs, action, done)
#             if done:
#                 result = self._report_result_upon_completion(goal_idx)
#                 break
#             if self.cfg.enable_render:
#                 self.env.render(mode="human")
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             if obs is None:
#                 obs = last_obs  # use cached observation in case of `None` observation
#             else:
#                 last_obs = obs  # cache valid observation
#             if self.cfg.goal_conditional == "onehot":
#                 goal = self.goal_fn(obs, goal_idx, i)
#             action = self._get_action(obs, goal=goal, sample=True)
#             action_history.append(action)
#         logging.info(f"Total reward: {total_reward}")
#         logging.info(f"Final info: {info}")
#         logging.info(f"Result: {result}")
#         return total_reward, action_history, info, result

#     def _report_result_upon_completion(self, goal_idx=None):
#         pass

#     def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
#         print(obs, chosen_action, done)
#         raise NotImplementedError

#     def _get_action(self, obs, goal=None, sample=True):
#         with utils.eval_mode(
#             self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
#         ):
#             obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
#             enc_obs = self.obs_encoding_net(obs).squeeze(0)
#             enc_obs = einops.repeat(
#                 enc_obs, "obs -> batch obs", batch=self.cfg.action_batch_size
#             )
#             # Now, add to history. This automatically handles the case where
#             # the history is full.
#             self.obs_context.append(enc_obs)
#             if self.cfg.goal_conditional:
#                 if self.cfg.goal_conditional == "onehot":
#                     # goal: Tensor[G]
#                     goal = torch.Tensor(goal).float().to(self.cfg.device)
#                     goal = einops.repeat(
#                         goal, "goal -> batch goal", batch=self.cfg.action_batch_size
#                     )
#                     self.goal_context.append(goal)
#                     goal_seq = torch.stack(tuple(self.goal_context), dim=0)
#                 elif self.cfg.goal_conditional == "future":
#                     # goal: Tensor[T, O], a slice of future observations
#                     goal = torch.Tensor(goal).float().to(self.cfg.device).unsqueeze(0)
#                     enc_goal = self.obs_encoding_net(goal).squeeze(0)
#                     enc_goal = einops.repeat(
#                         enc_goal,
#                         "seq goal -> seq batch goal",
#                         batch=self.cfg.action_batch_size,
#                     )
#                     # We are not using goal_context for this case, since
#                     # the future slice is already a sequence.
#                     goal_seq = enc_goal
#             else:
#                 goal_seq = None
#             if self.cfg.use_state_prior:
#                 enc_obs_seq = torch.stack(tuple(self.obs_context), dim=0)  # type: ignore
#                 action_latents = self.state_prior.generate_latents(
#                     enc_obs_seq,
#                     torch.ones_like(enc_obs_seq).mean(dim=-1),
#                     goal=goal_seq,
#                 )
#             else:
#                 action_latents = self.action_ae.sample_latents(
#                     num_latents=self.cfg.action_batch_size
#                 )
#             actions = self.action_ae.decode_actions(
#                 latent_action_batch=action_latents,
#                 input_rep_batch=enc_obs,
#             )
#             actions = actions.cpu().numpy()
#             # Take the last action; this assumes that SPiRL (CVAE) already selected the first action
#             actions = actions[:, -1, :]
#             # Now action shape is (batch_size, action_dim)
#             if sample:
#                 actions = self.sampler(actions)
#             return actions

#     def run(self):
#         rewards = []
#         infos = []
#         results = []
#         if self.cfg.lazy_init_models:
#             self._init_action_ae()
#             self._init_obs_encoding_net()
#             self._init_state_prior()
#         for i in range(self.cfg.num_eval_goals):
#             for j in range(self.cfg.num_eval_eps):
#                 reward, actions, info, result = self.run_single_episode(goal_idx=i)
#                 rewards.append(reward)
#                 infos.append(info)
#                 results.append(result)
#                 torch.save(actions, Path.cwd() / f"goal_{i}_actions_{j}.pth")
#         self.env.close()
#         logging.info(rewards)
#         return rewards, infos, results

#     @property
#     def snapshot(self):
#         return Path(self.cfg.load_dir or self.work_dir) / "snapshot.pt"

#     def load_snapshot(self):
#         keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
#         with self.snapshot.open("rb") as f:
#             payload = torch.load(f, map_location=self.device)
#         loaded_keys = []
#         for k, v in payload.items():
#             if k in keys_to_load:
#                 loaded_keys.append(k)
#                 self.__dict__[k] = v.to(self.cfg.device)

#         if len(loaded_keys) != len(keys_to_load):
#             raise ValueError(
#                 "Snapshot does not contain the following keys: "
#                 f"{set(keys_to_load) - set(loaded_keys)}"
#             )



class Workspace:
    def __init__(self, cfg):
        # self.work_dir = f"/media/disk4/wtyao/hyperpolicy/EXP/c_bet/{cfg.task_name}/{cfg.model_num}"
        self.work_dir = "/media/disk4/wtyao/baselines/cbet/play-to-policy/exp_local/2024.11.06/123835_kitchen_train"

        # self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        if self.cfg.goal_conditional:
            self.goal_fn = hydra.utils.instantiate(self.cfg.env.goal_fn, cfg)

        self.epoch = 0
        self.load_snapshot()

        # Set up rolling window contexts.
        self.window_size = cfg.window_size
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.window_size)

        self._setup_action_sampler()

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
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_action_sampler(self):
        def sampler(actions):
            idx = np.random.randint(len(actions))
            return actions[idx]

        self.sampler = sampler

    def run_single_episode(self, goal_idx=None):
        action_history = []
        obs = self.env.reset()
        last_obs = obs
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        if self.cfg.goal_conditional:
            goal = self.goal_fn(obs, goal_idx, 0)
        else:
            goal = None
        action = self._get_action(obs, goal=goal, sample=True)
        done = False
        total_reward = 0
        action_history.append(action)
        for i in range(self.cfg.num_eval_steps + 1):
            if self.cfg.plot_interactions:
                self._plot_obs_and_actions(obs, action, done)
            if done:
                result = self._report_result_upon_completion(goal_idx)
                break
            if self.cfg.enable_render:
                self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            if self.cfg.goal_conditional == "onehot":
                goal = self.goal_fn(obs, goal_idx, i)
            action = self._get_action(obs, goal=goal, sample=True)
            action_history.append(action)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        logging.info(f"Result: {result}")
        return total_reward, action_history, info, result

    def _get_action(self, obs, goal=None, sample=True):
        with utils.eval_mode(
            self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
        ):
            obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            enc_obs = self.obs_encoding_net(obs).squeeze(0)
            enc_obs = einops.repeat(enc_obs, "obs -> batch obs", batch=self.cfg.action_batch_size)
            # Now, add to history. This automatically handles the case where
            # the history is full.
            self.obs_context.append(enc_obs)
            
            goal = torch.Tensor(goal).float().to(self.cfg.device).unsqueeze(0)
            enc_goal = self.obs_encoding_net(goal).squeeze(0)
            enc_goal = einops.repeat(enc_goal,"seq goal -> seq batch goal",batch=self.cfg.action_batch_size,)
            # We are not using goal_context for this case, since
            # the future slice is already a sequence.
            goal_seq = enc_goal
            
            enc_obs_seq = torch.stack(tuple(self.obs_context), dim=0)  # type: ignore

            action_latents = self.state_prior.generate_latents(
                enc_obs_seq,
                torch.ones_like(enc_obs_seq).mean(dim=-1),
                goal=goal_seq,
            )
            
            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
                input_rep_batch=enc_obs,
            )
            actions = actions.cpu().numpy()
            # Take the last action; this assumes that SPiRL (CVAE) already selected the first action

            # actions = actions[:, -1, :] # changed
            
            # Now action shape is (batch_size, action_dim)
            if sample:
                actions = self.sampler(actions)
            return actions

    def run(self):
        rewards = []
        infos = []
        results = []
        if self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.num_eval_goals):
            for j in range(self.cfg.num_eval_eps):
                reward, actions, info, result = self.run_single_episode(goal_idx=i)
                rewards.append(reward)
                infos.append(info)
                results.append(result)
                torch.save(actions, Path.cwd() / f"goal_{i}_actions_{j}.pth")
        self.env.close()
        logging.info(rewards)
        return rewards, infos, results

    @property
    def snapshot(self):
        return Path(self.work_dir) / "snapshot.pt"

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.cfg.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
            
    def testing(self, cfg=None, env=None, num_episodes=100, device=torch.device("cuda")):
        obs_horizon = cfg.obs_horizon
        pickle_path = os.path.join(cfg.model_path, "stats.pickle")
        with open(pickle_path, "rb") as f:
            all_stats = pickle.load(f)
        tasks = [generate_key_from_value_corrected(value) for value in cfg.task_names]
        task_stats = {cap: orig for cap, orig in zip(tasks, cfg.task_names)}
        
        goal_img_idx = 226
        goal_image_path = f"/mnt/lv1/robomimic/mimicgen_videos/core/coffee_d0/0/{goal_img_idx}.png"
        goal_image = Image.open(goal_image_path).convert('RGB').resize((128, 128))
        # goal_image = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),])(goal_image).unsqueeze(0).to(device)
        
        suc_dict = {}
        for task_name in tasks:
            suc = 0
            tasks_list = []
            for seed in range(1000, 1050):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                imgs = []
                imgs_wrist = []
                
                options = {}
                # # Choose environment
                # options["env_name"] = choose_mimicgen_environment(task)
                options["env_name"] = task_name
                # # Choose robot
                # options["robots"] = choose_robots(exclude_bimanual=True)
                options["robots"] = 'Panda'
                # Load the desired controller
                options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
                
                env = configure_environment(cfg, options)
                # Reset the environment and set the camera
                obs = env.reset()
                env.viewer.set_camera(camera_id=0)

                # Process images    
                img_obs_deque = process_images(obs, 'agentview_image', cfg, obs_horizon)
                if cfg.use_wrist:
                    wrist_img_obs_deque = process_images(obs, 'robot0_eye_in_hand_image', cfg, obs_horizon)
                
                # get first observation
                max_steps = min(cfg.max_steps,600)
                # keep a queue of last 2 steps of observations
                obs_horizon = cfg.obs_horizon
                
                state_t = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']])
                obs_deque = collections.deque([state_t] * obs_horizon, maxlen=obs_horizon)
                start_image = torch.from_numpy(deepcopy(img_obs_deque[0])).unsqueeze(0).to(device, dtype=torch.float32).permute(0, 3, 1, 2)
                
                done = False
                step_idx = 0
                rewards = list()
                B = 1
                # track completion order
                while not done:
                    with torch.no_grad():
                        # stack the last obs_horizon (2) number of observations
                        obs_seq = np.stack(obs_deque)
                        obs_seq = normalize_data(obs_seq, stats=all_stats[task_stats[task_name]]["obs"])  # (T,obs)
                        obs_seq = torch.from_numpy(obs_seq).to(device, dtype=torch.float32).unsqueeze(0)
                        visual_seq = np.stack(img_obs_deque)
                        visual_seq = convert_images_to_tensors(visual_seq, None).to(device).unsqueeze(0)
                        # task_txt = [task_stats[task_name]]
   

                        # actions = self.inferece_forward(obs_seq, visual_seq, start_image, goal_image, task_txt)
                        # action_pred = unnormalize_data(actions.detach().to("cpu").numpy()[0], stats=all_stats[task_stats[task_name]]["actions"])
                        # start = obs_horizon - 1
                        # end = start + cfg.action_horizon
                        # action = action_pred[start:end, :]

                        action = self._get_action(obs_seq, goal=goal_image, sample=True)
                        action_pred = unnormalize_data(actions, stats=all_stats[task_stats[task_name]]["actions"])
                        
                        for i in range(len(action)):
                            # stepping env
                            obs, reward, done, info = env.step(action[i])
                            state_t = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']])
                            obs_deque.append(state_t)
                            
                            agentview_image = obs['agentview_image']
                            agentview_image = agentview_image[::-1, :, :]
                            raw_env_image = cv2.resize(agentview_image, cfg.bc_resize)
                            img_obs_deque.append(raw_env_image.copy())
                            imgs.append(raw_env_image)
                            rewards.append(reward)

                            if cfg.use_wrist:
                                wrist_image = obs['robot0_eye_in_hand_image']
                                wrist_image = wrist_image[::-1, :, :]
                                raw_wrist_image = cv2.resize(wrist_image, cfg.bc_resize)
                                wrist_img_obs_deque.append(raw_wrist_image.copy())
                                imgs_wrist.append(raw_wrist_image)

                            # update progress bar
                            step_idx += 1
                            print("seed", seed-1000, 'current_step', step_idx, 'reward', reward)
                            if reward > 0:
                                suc += 1
                                tasks_list.append(seed-1000)
                                done = True
                                break

                            if step_idx > max_steps:
                                done = True
                                      
                eval_save_path = os.path.join(cfg.model_path, f"{task_name}_{cfg.model_name}_goal{goal_img_idx}")
                os.makedirs(eval_save_path, exist_ok=True)
                
                # save eval gif
                video_save_path = os.path.join(eval_save_path, f"eval_{seed}_{env.reward()}.mp4")
                # imageio.mimsave(video_save_path, imgs)
                imageio.mimsave(video_save_path, imgs, fps=30, macro_block_size=None)
                print("*********save video***********")
                
                if cfg.use_wrist:
                    video_save_path = os.path.join(eval_save_path, f"eval_{seed}_wrist_{env.reward()}.mp4")
                    # imageio.mimsave(video_save_path, imgs_wrist)
                    imageio.mimsave(video_save_path, imgs_wrist, fps=30, macro_block_size=None)
                
            suc_string = str(suc) 
            with open(os.path.join(eval_save_path, f"one_task_result.txt"), 'w') as file:
                file.write(f"{suc_string}\n")
                list_string = ', '.join(map(str, tasks_list))
                file.write(list_string + '\n')
            suc_dict[task_name] = suc_string
            
        dict_str = "\n".join([f"{key}: {value}" for key, value in suc_dict.items()])
        with open(f'{cfg.model_path}/all_tasks_results.txt', 'a') as file:
            file.write(dict_str+"\n")