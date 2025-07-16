import torch
from models import Generator, TextEncoder
from utils import encode_text, vocab_size
import torchvision.utils as vutils
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Config
z_dim = 64
embed_dim = 64
img_size = 28 * 28
max_len = 64

# Load models
generator = Generator(z_dim, embed_dim, img_size).to(device)
text_encoder = TextEncoder(vocab_size, embed_dim).to(device)

generator.load_state_dict(torch.load("saved_models/generator.pth"))
text_encoder.load_state_dict(torch.load("saved_models/text_encoder.pth"))

generator.eval()
text_encoder.eval()

# Get input
arabic_text = input("📝 Enter Arabic sentence: ")
encoded = encode_text(arabic_text, max_len)
text_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

with torch.no_grad():
    text_embed = text_encoder(text_tensor)
    noise = torch.randn(1, z_dim).to(device)
    generated = generator(noise, text_embed).view(1, 1, 28, 28)

    os.makedirs("generated_output", exist_ok=True)
    output_path = "generated_output/generated_image.png"
    vutils.save_image(generated, output_path, normalize=True)
    print(f"✅ Generated image saved to: {output_path}")
