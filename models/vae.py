from torch import nn, Tensor

class Vae(nn.Module):
    def __init__(self,
                 encode_dims: list[int],
                 latent_dim: int,
                 decode_dims: list[int] = None):
        super(Vae, self).__init__()
        if decode_dims is None:
            decode_dims = list(reversed(encode_dims))
        
        # encoder
        in_dim = encode_dims[0]
        encoder_layers = []
        for d in encode_dims[1:]:
            encoder_layers.append(nn.Linear(in_dim, d))
            encoder_layers.append(nn.ReLU())
            in_dim = d
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.get_mu = nn.Linear(encode_dims[-1], latent_dim)
        self.get_logv = nn.Linear(encode_dims[-1], latent_dim)

        # decoder
        in_dim = latent_dim
        decoder_layers = []
        for d in decode_dims:
            decoder_layers.append(nn.Linear(in_dim, d))
            decoder_layers.append(nn.ReLU())
            in_dim = d
        self.decoder = nn.Sequential(*decoder_layers)


    def encode(self, x: Tensor) -> Tensor:
        out = self.encoder(x)
        mu = self.get_mu(out)
        logv = self.get_logv(out)
        return [mu, logv]


    def reparameterize(self, mu, logvar):
        pass


    def decode(self, z: Tensor) -> Tensor:
        out = self.decoder(z)
        return out
        

    def forward(self, x):
        pass
