import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import secrets

from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_dataset


root_dir = Path(__file__).parent


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    task = hydra.utils.instantiate(cfg.task)
    train_dataset, test_dataset = sample_dataset(task, **cfg.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    inputs, targets = next(iter(train_loader))
    input_size = inputs.shape[-1]
    output_size = targets.shape[-1]

    model = hydra.utils.instantiate(cfg.model)
    model = model(input_size=input_size, output_size=output_size)
    model = RNNLightningModule(model=model, **cfg.optimizer)

    logger = hydra.utils.instantiate(cfg.logger)
    logger = logger(config=OmegaConf.to_object(cfg))
    trainer = L.Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model, train_loader, test_loader)
    r2 = trainer.progress_bar_metrics.get('val_r2', np.nan)
    acc = trainer.progress_bar_metrics.get('val_acc', np.nan)

    model_params = model.model.state_dict()
    task_signal = cfg.task.signal
    task_target = cfg.task.target
    checkpoint_dir = root_dir / f"checkpoints/{task_signal}_{task_target}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving model to {checkpoint_dir.absolute()}")
    torch.save(model_params, checkpoint_dir / f"{secrets.token_hex(8)}-r2={r2:.2f}-acc={acc:.2f}.ckpt")


if __name__ == "__main__":
    main()