import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, latent_dim, nhead, num_layers, window_size, *args, **kwargs):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim * window_size, latent_dim)

    def forward(self, x):
        return self.fc(self.encoder(x).reshape(x.shape[0], -1))


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, latent_dim, nhead, num_layers, window_size, *args, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(latent_dim, embed_dim * window_size)
        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.decoder(self.fc(x).reshape(x.shape[0], self.window_size, -1))
