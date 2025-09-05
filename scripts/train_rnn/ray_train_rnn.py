import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import secrets

import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_dataset


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
    train_dataset, test_dataset = sample_dataset(task, **cfg_partial['dataset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg_partial['batch_size'], shuffle=True)
    inputs, targets = next(iter(train_loader))
    cfg_partial["input_size"] = inputs.shape[-1]
    cfg_partial["output_size"] = targets.shape[-1]
    cfg_partial["task_signal"] = task_signal
    cfg_partial["task_target"] = task_target
    del train_loader, inputs, targets

    def train_loop_per_worker(config):

        # Fetch training dataset.
        train_dataset_shard = ray.train.get_dataset_shard("train")
        test_dataset_shard = ray.train.get_dataset_shard("test")

        # Instantiate and prepare model for training.
        model = config["model"](input_size=config["input_size"], output_size=config["output_size"])
        model = RNNLightningModule(model=model, **config["optimizer"])
        model = ray.train.torch.prepare_model(model)

        # Define loss and optimizer.
        logger = config["logger"](config=OmegaConf.to_object(config))
        trainer = L.Trainer(**config["trainer"], logger=logger)

        # Create data loader.
        train_dataloader = train_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"], dtypes=torch.float
        )
        test_dataloader = test_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"], dtypes=torch.float
        )
        trainer.fit(model, train_dataloader, test_dataloader)
        r2 = trainer.progress_bar_metrics.get('val_r2', np.nan)
        acc = trainer.progress_bar_metrics.get('val_acc', np.nan)

        model_params = model.model.state_dict()
        task_signal = config["task_signal"]
        task_target = config["task_target"]
        checkpoint_dir = root_dir / f"checkpoints/{task_signal}_{task_target}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_params, checkpoint_dir / f"{secrets.token_hex(8)}-r2={r2:.2f}-acc={acc:.2f}.ckpt")

    # Define configurations.
    scaling_config = ScalingConfig(
        num_workers=4, 
        use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 0.25},
    )

    # Define datasets.
    train_dataset = ray.data.from_torch(train_dataset)
    test_dataset = ray.data.from_torch(test_dataset)
    datasets = {"train": train_dataset, "test": test_dataset}

    # Initialize the Trainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg_partial,
        scaling_config=scaling_config,
        # run_config=run_config,
        datasets=datasets
    )

    # Train the model.
    result = trainer.fit()

if __name__ == "__main__":
    main()