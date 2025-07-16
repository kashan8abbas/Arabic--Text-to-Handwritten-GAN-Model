import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import KHATTDataset
from models import Generator, Discriminator, TextEncoder
from utils import vocab_size
import torchvision.utils as vutils
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams
z_dim = 64
embed_dim = 64
img_size = 28 * 28
batch_size = 32
num_epochs = 200
lr = 3e-4

# Dataset
dataset = KHATTDataset("dataset/images", "dataset/texts")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
text_encoder = TextEncoder(vocab_size, embed_dim).to(device)
generator = Generator(z_dim, embed_dim, img_size).to(device)
discriminator = Discriminator(img_size, embed_dim).to(device)

opt_gen = optim.Adam(generator.parameters(), lr=lr)
opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

os.makedirs("saved_models", exist_ok=True)
os.makedirs("generated_samples", exist_ok=True)

for epoch in range(num_epochs):
    for batch_idx, (real, text_indices) in enumerate(loader):
        real = real.view(-1, img_size).to(device)
        text_indices = text_indices.to(device)
        text_embed = text_encoder(text_indices)

        # ========== Train Discriminator ==========
        noise = torch.randn(real.size(0), z_dim).to(device)
        fake = generator(noise, text_embed).detach()
        disc_real = discriminator(real, text_embed.detach()).view(-1)
        disc_fake = discriminator(fake, text_embed.detach()).view(-1)
        lossD = (criterion(disc_real, torch.ones_like(disc_real)) +
                 criterion(disc_fake, torch.zeros_like(disc_fake))) / 2

        discriminator.zero_grad()
        lossD.backward()
        opt_disc.step()

        # ========== Train Generator ==========
        fake = generator(noise, text_embed)
        output = discriminator(fake, text_embed).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        generator.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                  f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")


    # Save models every epoch
    torch.save(generator.state_dict(), "saved_models/generator.pth")
    torch.save(text_encoder.state_dict(), "saved_models/text_encoder.pth")

    # Save sample image
    with torch.no_grad():
        sample_noise = torch.randn(1, z_dim).to(device)
        sample_text = "المدرسة جميلة"  # Arabic: "The school is beautiful"
        from utils import encode_text
        encoded = encode_text(sample_text, max_len=64)
        text_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        text_embed = text_encoder(text_tensor)
        fake_img = generator(sample_noise, text_embed).view(1, 1, 28, 28)
        vutils.save_image(fake_img, f"generated_samples/sample_epoch_{epoch+1}.png", normalize=True)