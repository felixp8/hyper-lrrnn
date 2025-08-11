import hydra
import torch
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path

from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_dataset


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    task = hydra.utils.instantiate(cfg.task)
    train_dataset, test_dataset = sample_dataset(task, **cfg.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
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

    model_params = model.model.state_dict()
    checkpoint_dir = Path(f"results/{cfg.run_name}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_params, checkpoint_dir / "model.ckpt")


if __name__ == "__main__":
    main()