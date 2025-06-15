"""
Reference:
- https://github.com/debtanu177/CVAE_MNIST/blob/master/train_cvae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Convolutional VAE (~2M params) for 224×224 RGB images,
    with corrected ConvTranspose settings to restore exact spatial size.
    """
    def __init__(self, latent_dim=128, im_channels=3, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')

        # ─── Encoder ──────────────────────────────────────────────────────────────
        # conv1–conv5 progressively downsample 224→112→56→28→14→5
        self.conv1 = nn.Conv2d(im_channels, 16, kernel_size=5, stride=2, padding=2)  # 224→112
        self.conv2 = nn.Conv2d(16,         32, kernel_size=5, stride=2, padding=2)  # 112→56
        self.conv3 = nn.Conv2d(32,         64, kernel_size=5, stride=2, padding=2)  # 56→28
        self.conv4 = nn.Conv2d(64,         64, kernel_size=5, stride=2, padding=2)  # 28→14
        self.conv5 = nn.Conv2d(64,         64, kernel_size=5, stride=2, padding=0)  # 14→ 5

        # Flatten (64 channels × 5×5 spatial)
        self.flatten_size = 64 * 5 * 5  # =1600

        # Latent projections
        self.fc1    = nn.Linear(self.flatten_size, 384)
        self.mu     = nn.Linear(384, latent_dim)
        self.logvar = nn.Linear(384, latent_dim)

        # ─── Decoder ──────────────────────────────────────────────────────────────
        # FC to expand back to flattened features
        self.fc2 = nn.Linear(latent_dim, 384)
        self.fc3 = nn.Linear(384, self.flatten_size)

        # deconv1 inverts conv5: 5→14
        self.deconv1 = nn.ConvTranspose2d(
            64, 64,
            kernel_size=5, stride=2,
            padding=0,        # matches conv5 padding=0
            output_padding=1  # to achieve 14 instead of 13
        )
        # deconv2 inverts conv4: 14→28
        self.deconv2 = nn.ConvTranspose2d(
            64, 64,
            kernel_size=5, stride=2,
            padding=2,        # matches conv4 padding=2
            output_padding=1  # to achieve 28 exactly
        )
        # deconv3 inverts conv3: 28→56
        self.deconv3 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=5, stride=2,
            padding=2,
            output_padding=1
        )
        # deconv4 inverts conv2: 56→112
        self.deconv4 = nn.ConvTranspose2d(
            32, 16,
            kernel_size=5, stride=2,
            padding=2,
            output_padding=1
        )
        # deconv5 inverts conv1: 112→224
        self.deconv5 = nn.ConvTranspose2d(
            16, im_channels,
            kernel_size=5, stride=2,
            padding=2,
            output_padding=1
        )

    def encode(self, x: torch.Tensor):
        x = x.to(self.device)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))       # now (B,64,5,5)
        h = h.view(x.size(0), -1)       # flatten to (B,1600)
        h = F.relu(self.fc1(h))         # project to hidden
        mu     = self.mu(h)             # (B, latent_dim)
        logvar = self.logvar(h)         # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = F.relu(self.fc2(z))                           # (B,384)
        h = F.relu(self.fc3(h))                           # (B,1600)
        h = h.view(z.size(0), 64, 5, 5)                   # reshape to feature map
        h = F.relu(self.deconv1(h))                       # (B,64,14,14)
        h = F.relu(self.deconv2(h))                       # (B,64,28,28)
        h = F.relu(self.deconv3(h))                       # (B,32,56,56)
        h = F.relu(self.deconv4(h))                       # (B,16,112,112)
        recon = torch.sigmoid(self.deconv5(h))            # (B,3,224,224)
        return recon

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=128, im_channels=3, device=device).to(device)
    summary(model, input_size=(3, 224, 224), device=str(device))
    print("VAE (~2M params) loaded successfully.")

    model.eval()

    # 더미 배치 생성 (예: B=4)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)  # (B, 3, 224, 224)

    # 순전파
    with torch.no_grad():
        recon, mu, logvar = model(x)

    print("Input shape    :", x.shape)
    print("Reconstruction :", recon.shape)  # 기대: torch.Size([4, 3, 224, 224])
    print("Mu shape       :", mu.shape)     # torch.Size([4, latent_dim])
    print("Logvar shape   :", logvar.shape) # torch.Size([4, latent_dim])
