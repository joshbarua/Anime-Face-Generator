import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from torchsummary import summary

# number of gpu available 
ngpu = 1

# device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device = torch.device("cpu")

# number of training epochs 
num_epochs = 3

# batch size during training
batch_size = 128

# size of z latent vector
nz = 100

# learning rate for optimizer
learning_rate = 0.0002

# number of GPUs available, use 0 for CPU mode
ngpu = 1

# resize images to fixed size, convert image to tensor, normalize pixel image from [0, 255] to [-1, 1]
train_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])

# import data and apply transformations
train_data = datasets.ImageFolder(root='~joshbarua/DLProjects/Anime-Face-Generator/data', transform=train_transform)

# create iterable over faces dataset
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# plot training images
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization function for generator and discriminator layers
def weights_init(m):
    classname = m.__class__.__name__

    # convolutional layers weights initialized from normal dist with mean: 0, std: .02
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    # batch-normalization layer weights initialized from normal dist with mean: 1, std: .02, and 0 bias
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)  

# generator network 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            # block 1: input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),

            # block 2: input is (64 * 8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),

            # block 3: input is (64 * 4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),

            # block 4: input is (64 * 2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # block 5: input is (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()

            # output: output is (3) x 64 x 64
        )
    
    # forward function which is fed the noise vector
    def forward(self, input):
        output = self.main(input)
        return output

# discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # block 1: input is (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # block 2: input is (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # block 3: input is (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # block 4: input is (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # block 5: input is (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()

            # output: 1
        )

    # forward function which is fed generated image and predicts (1) real or (0) fake
    def forward(self, input):
        output = self.main(input)
        return output

# move networks to device and intialize parametric layers
generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
summary(generator, (100, 64, 64))
summary(discriminator, (3, 64, 64))

# loss function
adversarial_loss = nn.BCELoss()

# create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# generator loss function which is fed discriminator output (when fed generated-produced images) and ground truth label (1)
def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss

# discriminator loss function which is called twice:
# once with real (original images) output predictions and label 1 
# again with fake (generated images) output predictions and label 0
def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

# intialize optimizer
G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))

#lists to keep track of progress 
img_list = []
G_loss_list = []
D_loss_list = []
iters = 0

# training the networks 
print("Starting Training Loop...")
for epoch in range(num_epochs): 
    
    # iterate through images in dataset
    for index, data in enumerate(train_loader, 0):
        print(iters)

        D_optimizer.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        
        real_target = Variable(torch.ones(b_size).to(device))
        fake_target = Variable(torch.zeros(b_size).to(device))

        # feed discriminator the real images
        output = discriminator(real_images)
        real_target = real_target.view(-1,1)
        D_real_loss = discriminator_loss(output, real_target)
        D_real_loss.backward()

        # sample noise vectors
        noise_vector = torch.randn(b_size, nz, 1, 1, device=device) 

        # pass noise vectors through the generator
        generated_images = generator(noise_vector)

        # feed fake (generated) images to the discriminator
        output = discriminator(generated_images.detach())
        fake_target = fake_target.view(-1,1)
        D_fake_loss = discriminator_loss(output,fake_target)
        D_fake_loss.backward()

        # optimize discriminator with total loss from fake and real images
        D_total_loss = D_real_loss + D_fake_loss
        D_loss_list.append(D_total_loss)
        D_optimizer.step()

        # generated images from earlier are passed to optimized (updated parameters) discriminator network
        G_optimizer.zero_grad()
        gen_output = discriminator(generated_images)

        # calculate loss and optimize generator parameters
        G_loss = generator_loss(gen_output, real_target)
        G_loss_list.append(G_loss)
        G_loss.backward()
        G_optimizer.step()

        # check how the generator is doing by saving generated output on fixed noise
        if (iters % 250 == 0) or ((epoch == num_epochs-1) and (index == len(train_loader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

print("done")
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()




