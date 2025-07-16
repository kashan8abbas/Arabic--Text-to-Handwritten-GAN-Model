import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):  # x: [batch, seq_len]
        emb = self.embed(x)
        _, (hidden, _) = self.lstm(emb)
        return hidden.squeeze(0)  # [batch, hidden_dim]

class Generator(nn.Module):
    def __init__(self, z_dim, text_dim, img_channels=1, feature_maps=64):
        super().__init__()
        self.input_dim = z_dim + text_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, feature_maps * 4, 4, 1, 0),  # 4x4
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1),  # 32x32
            nn.Tanh()
        )

    def forward(self, z, text_embed):
        x = torch.cat([z, text_embed], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, text_dim=64, feature_maps=64):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 32 * 32)

        self.net = nn.Sequential(
            nn.Conv2d(2, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, text_embed):
        proj_text = self.text_proj(text_embed).view(-1, 1, 32, 32)
        x = torch.cat([img, proj_text], dim=1)
        return self.net(x)
