import os
import pathlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from GAN import Discriminator, Generator, save_GAN_images, train
from utils.load_save import load_config

# Get the current date and time
now = datetime.now()
Date = now.strftime("%Y-%m-%d")
Time = now.strftime("%H-%M-%S")

# Create the directory name
when = f"{Date}_{Time}"

config = load_config(
    config_path=pathlib.Path(
        (os.path.join(os.path.expanduser("~/dx"), "config/config.toml"))
    )
)

latent_dim = 100
n_epochs = 100

device = torch.device(
    f"cuda:{int(config.nvidia)}" if torch.cuda.is_available() else "cpu"
)

batch_size = config.batch_size

root_dir = os.getcwd()

train_dir = os.path.join(
    root_dir, "data", config.which_data, config.train_data
)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
# print(train_data)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

# print(train_loader.dataset)

sample_x, _ = next(iter(train_loader))
print(sample_x.shape)


netD = Discriminator(sample_x).to(device)
netG = Generator(sample_x, latent_dim).to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.002)
criterion = nn.BCELoss()

print("初期状態")
save_GAN_images(netG, latent_dim, device, 0, when)
train(
    netD,
    netG,
    optimD,
    optimG,
    criterion,
    n_epochs,
    train_loader,
    latent_dim,
    device,
    batch_size,
    when,
)
