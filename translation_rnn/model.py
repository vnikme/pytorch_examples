from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image

torch.manual_seed(1)


class Seq2SeqVAE(nn.Module):
    def __init__(self, sp, embedding_size, hidden_size, n_layers, latent_size, dropout, device):
        super(Seq2SeqVAE, self).__init__()

        # parameters
        dict_size = sp.GetPieceSize()
        self.dict_size = dict_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size

        # encoder
        self.embedding = nn.Embedding(dict_size, embedding_size).to(device)
        self.input_projection = torch.nn.Linear(embedding_size, hidden_size).to(device)
        self.encoder = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout).to(device)
        self.encoder_mu = torch.nn.Linear(hidden_size, latent_size).to(device)
        self.encoder_logvar = torch.nn.Linear(hidden_size, latent_size).to(device)

        # decoder
        self.decoder_projection = torch.nn.Linear(latent_size, hidden_size).to(device)
        self.decoder0 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout).to(device)
        self.decoder1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout).to(device)
        self.output_projection = torch.nn.Linear(hidden_size, dict_size).to(device)


    def embed(self, data):
        return self.input_projection(self.embedding(data))

    def encode(self, data, lengths, h):
        output, _ = self.encoder(data, h)
        x = []
        for i in range(output.shape[0]):
            x.append(output[i,lengths[i] - 1,:])
        x = torch.stack(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def latent_2_hidden(self, z):
        return torch.stack([self.decoder_projection(z)] * self.n_layers)

    def decode(self, data, z, idx):
        if idx == 0:
            output, hidden = self.decoder0(data, z)
        else:
            output, hidden = self.decoder1(data, z)
        output = self.output_projection(output)
        return output, hidden

