import torch
from torch import nn, Tensor

class TextToImageVAE(nn.Module):
    def __init__(self,
                 encode_dims: list[int],
                 latent_dim: int,
                 decode_dims: list[int] | None = None,
                 eps: Tensor | None = None):
        super(TextToImageVAE, self).__init__()
        if decode_dims is None:
            decode_dims = list(reversed(encode_dims))
        self.eps = eps
        
        # encoder
        in_dim = encode_dims[0]
        encoder_layers = []
        for dim in encode_dims[1:]:
            encoder_layers.append(nn.Sequential(
                nn.Conv1d(in_dim, dim, 3, 2),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ))
            in_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.get_mu = nn.Linear(encode_dims[-1], latent_dim)
        self.get_logv = nn.Linear(encode_dims[-1], latent_dim)

        # decoder
        in_dim = latent_dim
        decoder_layers = []
        for dim in decode_dims:
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim, dim, 3, 2),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ))
            in_dim = dim
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> list[Tensor]:
        out = self.encoder(x)
        mu = self.get_mu(out)
        logv = self.get_logv(out)
        return [mu, logv]

    def reparameterize(self, mu: Tensor, logv: Tensor) -> Tensor:
        std = torch.exp(0.5 * logv)
        eps = self.eps if self.eps else torch.rand_like(std)
        z = mu + eps * std
        return [z, eps]

    def decode(self, z: Tensor) -> Tensor:
        out = self.decoder(z)
        return out

    def forward(self, x: Tensor) -> list[Tensor]:
        mu, logv = self.encode(x)
        z, eps = self.reparameterize(mu, logv)
        xr = self.decode(z)
        return [xr, mu, logv, eps]
