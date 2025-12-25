import os
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import secrets

import ray
import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_ray_dataset


root_dir = Path(__file__).parent


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    ray.init(
        # _temp_dir=str((root_dir / "ray").absolute()),
    )

    task_signal = cfg.task.signal
    task_target = cfg.task.target
    cfg_partial = OmegaConf.to_object(hydra.utils.instantiate(cfg))
    task = cfg_partial['task']
    train_dataset, test_dataset = sample_ray_dataset(task, **cfg_partial['dataset'])
    data = train_dataset.take(1)
    inputs = np.stack([item["input"] for item in data], axis=0)
    targets = np.stack([item["target"] for item in data], axis=0)
    cfg_partial["input_size"] = inputs.shape[-1]
    cfg_partial["output_size"] = targets.shape[-1]
    cfg_partial["task_signal"] = task_signal
    cfg_partial["task_target"] = task_target
    del inputs, targets

    def train_loop_per_worker(config):
        run_token = secrets.token_hex(8)
        run_name = config["run_name"]

        # Fetch training dataset.
        train_dataset_shard = ray.train.get_dataset_shard("train")
        test_dataset_shard = ray.train.get_dataset_shard("test")

        # Instantiate and prepare model for training.
        model = config["model"](input_size=config["input_size"], output_size=config["output_size"])
        model = RNNLightningModule(model=model, **config["optimizer"])

        # Define loss and optimizer.
        logger = config["logger"](name=f"{run_name}-{run_token}", config=config)
        trainer = L.Trainer(
            **config["trainer"], 
            logger=logger,
            # strategy=ray.train.lightning.RayDDPStrategy(),
            # plugins=[ray.train.lightning.RayLightningEnvironment()],
            # callbacks=[ray.train.lightning.RayTrainReportCallback()],
        )
        # trainer = ray.train.lightning.prepare_trainer(trainer)

        # Create data loader.
        train_dataloader = train_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"]
        )
        test_dataloader = test_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"]
        )
        trainer.fit(model, train_dataloader, test_dataloader)
        r2 = trainer.progress_bar_metrics.get('val_r2', np.nan)
        acc = trainer.progress_bar_metrics.get('val_acc', np.nan)

        model_params = model.model.state_dict()
        task_signal = config["task_signal"]
        task_target = config["task_target"]
        checkpoint_dir = root_dir / f"checkpoints/{task_signal}_{task_target}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_params, checkpoint_dir / f"{run_token}-r2={r2:.2f}-acc={acc:.2f}.ckpt")

    # Define configurations.
    scaling_config = ScalingConfig(
        num_workers=len(os.sched_getaffinity(0)) // 2 - 1, 
        use_gpu=False,
        resources_per_worker={"CPU": 2, "GPU": 0.0},
    )

    # Define datasets.
    datasets = {"train": train_dataset, "test": test_dataset}
    dataset_config = ray.train.DataConfig(
        datasets_to_split=[],
    )

    # Initialize the Trainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg_partial,
        scaling_config=scaling_config,
        # run_config=run_config,
        datasets=datasets,
        dataset_config=dataset_config,
    )

    # Train the model.
    result = trainer.fit()

if __name__ == "__main__":
    main()
