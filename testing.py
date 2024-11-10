import hydra
import joblib
import logging
import numpy as np
from omegaconf import OmegaConf


@hydra.main(version_base="1.2", config_path="configs", config_name="eval_kitchen_future_cond")
def main(cfg):
    # Needs _recursive_: False since we have more objects within that we are instantiating
    # without using nested instantiation from hydra
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = "coffee_d0"
    cfg.task_num = "140828"
    workspace = hydra.utils.instantiate(cfg.env.workspace, cfg=cfg, _recursive_=False)

    cfg.obs_horizon = 2
    cfg.model_path = workspace.work_dir
    cfg.task_names = [cfg.task_name]
    cfg.use_wrist = False
    cfg.max_steps = 600
    

    device = torch.device("cuda:1")

    _ = workspace.testing(cfg, device=device)

    # rewards, infos, results = workspace.run()



    logging.info("==== Summary ====")
    logging.info(rewards)
    logging.info(infos)
    logging.info(results)
    logging.info(f"Average reward: {np.mean(rewards)}")
    logging.info(f"Std: {np.std(rewards)}")
    logging.info(f"Average result: {np.mean(results)}")
    logging.info(f"Std: {np.std(results)}")


if __name__ == "__main__":
    main()
