import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # ---- 编码器 ----
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc_mu    = nn.Linear(hidden_dim, latent_dim)   # 均值 μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log(σ²)
        # ---- 解码器 ----
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = μ + ε·σ,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))       # 输出像素概率

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    ELBO 负值:
      L = ReconLoss + β · KL(q(z|x) || p(z))
    ReconLoss 用逐元素 BCE（sum reduction）
    KL 有闭式解: -0.5 · Σ(1 + logσ² - μ² - σ²)
    """
    x = x.view(-1, 784)
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld