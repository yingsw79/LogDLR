import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae import AE
from ..modules.grl import WarmStartGradientReverseLayer


class LogDLR(AE):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grl = WarmStartGradientReverseLayer()
        input_dim = kwargs['latent_dim']
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, kwargs['output_dim']),
        )

    def forward(self, x):
        output = super().forward(x)
        output['pred'] = self.discriminator(self.grl(output['z']))
        return output

    @staticmethod
    def domain_loss(pred, label):
        return F.cross_entropy(pred, label.long().cuda())

    def loss_function(self, x, output, domain_label):
        return super().loss_function(x, output) + self.domain_loss(output['pred'], domain_label) * self.hparams.beta

    def training_step(self, batch, *args, **kwargs):
        (x_S, _), (x_T, _) = batch
        x = torch.cat((x_S, x_T))
        output = self.forward(x)
        domain_label = torch.cat((torch.zeros(x_S.shape[0]), torch.ones(x_T.shape[0])))
        loss = self.loss_function(x, output, domain_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        (x, _), i = batch
        output = self.forward(x)
        domain_label = torch.tensor([i] * x.shape[0])
        loss = self.loss_function(x, output, domain_label)
        self.log('val_loss', loss)
