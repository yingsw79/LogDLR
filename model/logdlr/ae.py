import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from sklearn import metrics
from ..modules.transformer import TransformerEncoder, TransformerDecoder
from transformers.optimization import get_cosine_schedule_with_warmup


class AE(L.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.encoder = TransformerEncoder(**kwargs)
        self.decoder = TransformerDecoder(**kwargs)

    def forward(self, x):
        z = self.encoder(x)
        return {'z': z, 'recon_x': self.decoder(z)}

    @staticmethod
    def recon_loss(recon_x, x):
        return F.mse_loss(recon_x, x, reduction='none').sum(dim=(1, 2))

    @staticmethod
    def embedding_loss(z):
        return 0.5 * torch.linalg.norm(z, dim=-1)**2

    def loss_function(self, x, output):
        loss = self.recon_loss(output['recon_x'], x) + self.embedding_loss(output['z']) * self.hparams.alpha
        return loss.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

    def _step(self, batch):
        x, _ = batch
        output = self.forward(x)
        loss = self.loss_function(x, output)
        return loss

    def training_step(self, batch, *args, **kwargs):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self._step(batch)
        self.log('val_loss', loss)

    def on_predict_start(self):
        self.recon_losses, self.labels = [], []

    def predict_step(self, batch, *args, **kwargs):
        x, label = batch
        output = self.forward(x)
        self.recon_losses.extend(self.recon_loss(output['recon_x'], x).tolist())
        self.labels.extend(label.tolist())

    def on_predict_end(self):
        self.th = self.threshold()

    def on_test_start(self):
        self.on_predict_start()

    def test_step(self, *args, **kwargs):
        self.predict_step(*args, **kwargs)

    def on_test_end(self):
        preds = [0 if re <= self.th else 1 for re in self.recon_losses]
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(self.labels, preds, average='binary')
        print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}.')

    def threshold(self, n=100):
        mean_N = np.mean([re for re, label in zip(self.recon_losses, self.labels) if label == 0])
        mean_A = np.mean([re for re, label in zip(self.recon_losses, self.labels) if label == 1])
        d = (mean_A - mean_N) / n
        res, max_f1 = 0, -1
        th = mean_N
        for _ in range(n):
            f1 = metrics.f1_score(self.labels, [0 if re <= th else 1 for re in self.recon_losses], average='binary')
            if f1 > max_f1:
                res = th
                max_f1 = f1
            th += d
        return res
