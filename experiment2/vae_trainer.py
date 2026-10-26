import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae import VAE, vae_loss


def train(epochs=10, batch_size=128, lr=1e-3, latent_dim=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 数据 ----------
    tfm = transforms.ToTensor()          # [0,1] 归一化，与 sigmoid 输出对应
    train_set = datasets.MNIST('./data', train=True,  download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    # ---------- 模型 & 优化器 ----------
    model = VAE(latent_dim=latent_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)

    # ---------- 训练循环 ----------
    model.train()
    n = len(train_set)
    for epoch in range(1, epochs + 1):
        total_loss = total_rec = total_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()

            recon_x, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon_x, x, mu, logvar)

            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_rec  += rec.item()
            total_kl   += kl.item()

        print(f"Epoch {epoch:2d} | "
              f"Loss {total_loss/n:.2f} | "
              f"Recon {total_rec/n:.2f} | "
              f"KL {total_kl/n:.2f}")

    torch.save(model.state_dict(), 'vae.pth')
    return model


@torch.no_grad()
def sample(model, num=16, latent_dim=20, device='cpu'):
    """从先验 p(z) 采样生成新图像（.npy 便于查看）"""
    model.eval()
    z = torch.randn(num, latent_dim, device=device)
    imgs = model.decode(z).view(-1, 1, 28, 28).cpu()
    import torchvision
    torchvision.utils.save_image(imgs, 'samples.png', nrow=4)
    print("已保存 samples.png")


if __name__ == '__main__':
    model = train(epochs=100)
    # 加载
    model = VAE()                          # 需要先手动构建相同结构
    model.load_state_dict(torch.load('vae.pth'))
    model.eval()
    sample(model, device=next(model.parameters()).device)