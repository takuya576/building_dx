import os

import matplotlib.pyplot as plt
import torch
import torchvision

# from IPython.display import display
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Discriminator(nn.Module):
    def __init__(self, sample_x):
        self.c, self.w, self.h = sample_x.shape[1:]
        self.image_size = self.c * self.w * self.h
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.net(x)
        return y


class Generator(nn.Module):
    def __init__(self, sample_x, latent_dim):
        self.c, self.w, self.h = sample_x.shape[1:]
        self.image_size = self.c * self.w * self.h
        super().__init__()
        self.net = nn.Sequential(
            self._linear(latent_dim, 128),
            self._linear(128, 256),
            self._linear(256, 512),
            nn.Linear(512, self.image_size),
            nn.Sigmoid(),
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, self.c, self.w, self.h)
        return y


# ノイズを生成する関数
def make_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, device=device)


def save_GAN_images(
    netG, latent_dim, device, epoch, when, n_rows=1, n_cols=5, size=224
):
    z = make_noise(n_rows * n_cols, latent_dim, device)
    images = netG(z)
    images = transforms.Resize(size)(
        images
    )  # 画像の短辺がsizeに揃うようにリサイズ
    img = torchvision.utils.make_grid(
        images, nrow=n_cols
    )  # 画像をn_cols個ずつ並べる
    img = transforms.functional.to_pil_image(img)  # tensor to pil_image
    # display(img)
    plt.imshow(img)
    result_image_dir = os.path.join(
        os.path.expanduser("~/dx/GAN/result/"), when
    )
    os.makedirs(result_image_dir, exist_ok=True)
    plt.savefig(os.path.join(result_image_dir, f"result{epoch}.png"))
    plt.show()


def train(
    netD,
    netG,
    optimD,
    optimG,
    criterion,
    n_epochs,
    dataloader,
    latent_dim,
    device,
    batch_size,
    when,
):
    # 学習モード
    netD.train()
    netG.train()
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    for epoch in range(1, n_epochs + 1):
        for x, _ in dataloader:
            if len(x) != batch_size:
                continue
            x = x.to(device)

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            # 識別器の学習
            z = make_noise(batch_size, latent_dim, device)
            fake = netG(z)
            pred_fake = netD(fake.detach())
            pred_real = netD(x.detach())
            loss_fake = criterion(
                pred_fake, fake_labels
            )  # ひどい偽物画像は、正しいと学習して欲しくないので、偽物ラベルは0のまま
            loss_real = criterion(
                pred_real, real_labels * 0.9
            )  # 生成器の生成タスクを軽くするために、本物画像は0.9のラベルを与える、学習しやすくなる
            # loss_real = criterion(pred_real, real_labels)
            lossD = loss_fake + loss_real
            lossD.backward()
            optimD.step()

            # 生成器の学習
            fake = netG(z)
            pred = netD(fake)
            lossG = criterion(pred, real_labels)
            lossG.backward()
            optimG.step()

        print(f"epoch: {epoch}, lossD: {lossD.item()}, lossG: {lossG.item()}")
        # 生成器の出力を保存
        if epoch % 1 == 0:
            save_GAN_images(netG, latent_dim, device, epoch, when)
