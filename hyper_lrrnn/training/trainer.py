import lightning as L
import torch
import functools
from sklearn.metrics import r2_score

from hyper_lrrnn.training.loss import lin_reg_loss, lin_reg_r2, accuracy, lin_reg_acc
from hyper_lrrnn.rnn import LowRankRNNWithReadout, FullRankRNNWithReadout, MixtureLowRankRNNWithReadout


class RNNLightningModule(L.LightningModule):
    def __init__(self, model, init_lr=1e-3, weight_decay=1e-4, regularizer=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'regularizer'])
        self.model = model
        self.regularizer = regularizer

        if isinstance(model, (LowRankRNNWithReadout, FullRankRNNWithReadout, MixtureLowRankRNNWithReadout)):
            self.loss_fn = torch.nn.functional.mse_loss
            self.r2_fn = lambda x, y: r2_score(
                x.cpu().detach().numpy().reshape(-1, x.shape[-1]),
                y.cpu().detach().numpy().reshape(-1, y.shape[-1]),
            )
            self.acc_fn = lambda x, y: accuracy(x, y, threshold=0.5, window=10)
        else:
            self.loss_fn = functools.partial(lin_reg_loss, alpha=0.01)
            self.r2_fn = functools.partial(lin_reg_r2, alpha=0.01)
            self.acc_fn = functools.partial(lin_reg_acc, alpha=0.01, threshold=0.5, window=10)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            inputs = batch["input"]
            target = batch["target"]
        else:
            inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss)
        if self.regularizer is not None:
            reg_loss = self.regularizer(self.model)
            loss = loss + reg_loss
            self.log('reg_loss', reg_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            inputs = batch["input"]
            target = batch["target"]
        else:
            inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log('val_loss', loss)
        metric = self.r2_fn(output, target)
        self.log('val_r2', metric, prog_bar=True, on_step=False, on_epoch=True)
        metric = self.acc_fn(output, target)
        self.log('val_acc', metric, prog_bar=True, on_step=False, on_epoch=True)
        if hasattr(self.model, "_participation_ratio"):
            with torch.no_grad():
                pr = self.model._participation_ratio()
            self.log('participation_ratio', pr, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.init_lr, weight_decay=self.hparams.weight_decay)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, min_lr=1e-6),
            #     "monitor": "val_loss",
            #     # "frequency": "indicates how often the metric is updated",
            # },
        }