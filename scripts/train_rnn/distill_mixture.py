import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
import secrets

from hyper_lrrnn.training.trainer import DistillLightningModule
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.rnn import LowRankRNNWithReadout


root_dir = Path(__file__).parent
chkpt_path = root_dir / "results/run-2025-12-26_09-43-18/hydra_logs/local/freq_1_temp.ckpt"  # hardcoded

@hydra.main(config_path="config", config_name="distill", version_base="1.1")
def main(cfg):
    task = hydra.utils.instantiate(cfg.task)
    train_dataset, test_dataset = sample_dataset(task, **cfg.dataset)
    task_input_size = train_dataset.dataset.tensors[0].shape[-1]
    task_output_size = train_dataset.dataset.tensors[1].shape[-1]

    # set up data for distillation
    ref_model = LowRankRNNWithReadout( # hardcoded
        input_size=task_input_size, output_size=task_output_size, 
        alpha=0.1, activation="tanh",
        rank=5, hidden_size=500,
        orth_gain=1.0, pr_gain=1.0,
        dropout_rate=0.0,
    )

    state_dict = torch.load(chkpt_path, map_location="cpu")
    ref_model.load_state_dict(state_dict)
    params = torch.cat([ref_model.m, ref_model.n, ref_model.I], axis=1)  # (N, r + r + I)

    full_input, full_target = train_dataset.dataset.tensors
    with torch.no_grad():
        full_model_output = super(LowRankRNNWithReadout, ref_model).forward(full_input)
        span = torch.cat([ref_model.m, ref_model.I], axis=1)  # (N, r + I)
        full_model_output = (torch.linalg.pinv(span) @ full_model_output.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, r + I)
    dataset = torch.utils.data.TensorDataset(full_input, full_model_output, full_target)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    inputs, targets = next(iter(train_loader))[:2]
    input_size = inputs.shape[-1]
    output_size = targets.shape[-1]

    model = hydra.utils.instantiate(cfg.model)
    model = model(input_size=input_size, output_size=output_size)
    model = DistillLightningModule(model=model, params=params, **cfg.optimizer)

    logger = hydra.utils.instantiate(cfg.logger)
    if callable(logger):
        logger = logger(config=OmegaConf.to_object(cfg))
    trainer = L.Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model, train_loader, test_loader)
    # r2 = trainer.progress_bar_metrics.get('val_r2', np.nan)
    # acc = trainer.progress_bar_metrics.get('val_acc', np.nan)

    model_params = model.model.state_dict()
    task_signal = cfg.task.signal
    task_target = cfg.task.target
    torch.save(model_params, f"{task_signal}_{task_target}_temp.ckpt")
    # checkpoint_dir = root_dir / f"checkpoints/{task_signal}_{task_target}"
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # print(f"saving model to {checkpoint_dir.absolute()}")
    # torch.save(model_params, checkpoint_dir / f"{secrets.token_hex(8)}-r2={r2:.2f}-acc={acc:.2f}.ckpt")


if __name__ == "__main__":
    main()
