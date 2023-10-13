from torch import nn

class Vae(nn.module):
    def __init__(self):
        super(Vae, self).__init__()

    def encode(self, x):
        pass

    def reparameterize(self, mu, logvar):
        pass

    def decode(self, z):
        pass

    def forward(self, x):
        pass
