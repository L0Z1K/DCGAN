import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

image_train = datasets.ImageFolder(root="data_faces/",
                              transform=transforms.Compose([
                                    transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                              ]))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.reshape([x.size(0), 100, 1, 1])
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False), # 128*32*32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #256*16*16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #512*8*8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), #1024*4*4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False), # 1*1*1
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.model(x)
        x = x.reshape([x.size(0), 1])
        return x

batch_size= 128
data_train = DataLoader(dataset=image_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator().to(device)
G.apply(weights_init)
D = Discriminator().to(device)
D.apply(weights_init)

optim_G = torch.optim.Adam(G.parameters(), lr=0.001)
optim_D = torch.optim.Adam(D.parameters(), lr=0.001)
criterion = nn.BCELoss()

start = time.time()
print("[+] Train Start")
total_epochs = 100
total_batch = len(data_train)
for epoch in range(total_epochs):
    avg_cost = [0, 0]
    for x, _ in data_train:
        x = x.to(device)

        z = torch.randn(batch_size, 100, device=device)

        z_img = G(z)

        real = (torch.FloatTensor(x.size(0), 1).fill_(1.0)).to(device)
        fake = (torch.FloatTensor(x.size(0), 1).fill_(0.0)).to(device)
        
        # Train Generator
        optim_G.zero_grad()
        g_cost = criterion(D(z_img), real)
        g_cost.backward()
        optim_G.step()

        z_img = z_img.detach().to(device)
        # Train Discriminator
        optim_D.zero_grad()
        d_cost = criterion(D(torch.cat((x, z_img))), torch.cat((real, fake)))
        d_cost.backward()
        optim_D.step()

        avg_cost[0] += g_cost
        avg_cost[1] += d_cost
    
    avg_cost[0] /= total_batch
    avg_cost[1] /= total_batch

    print("Epoch: %d, Generator: %f, Discriminator: %f"%(epoch+1, avg_cost[0], avg_cost[1]))
    z = torch.randn(36, 100, device=device)
    z_img = G(z)
    img_grid = make_grid(z_img, nrow=6, normalize=True)
    save_image(img_grid, "result/%d.png"%(epoch+1))
    end = time.time()
    total = int(end-start)
    print("[+] Train Time : %dm %ds"%(total//60, total%60))
