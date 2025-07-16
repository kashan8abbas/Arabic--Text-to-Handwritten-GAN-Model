import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):  
        emb = self.embed(x) 
        return emb.mean(dim=1)  

class Generator(nn.Module):
    def __init__(self, z_dim, text_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + text_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, z, text_embed):
        x = torch.cat([z, text_embed], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim, text_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + text_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embed):
        x = torch.cat([img, text_embed], dim=1)
        return self.net(x)
