import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim) 
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim) # Why we have this layer?
        
        # hidden_dim --> latent_dim
        # `mean` and `log_var` have `latent_dim`, so that `z` can have `latent_dim`.
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        
        # hidden_dim --> output_dim
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        
class Model(nn.Module):
    '''
    The input and output of the model are all flattened images. 
    '''
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, std_var):
        epsilon = torch.randn_like(std_var).to(DEVICE)        # sampling epsilon. Ensuring that ϵϵ has the same shape as the variance and is on the correct computing device.      
        z = mean + std_var*epsilon                          # reparameterization trick. Calculate zz using the provided mean and variance. 
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        
        # `log_var` == 2 log σ, where `σ` is the standard deviation of the distribution.
        # So, e^(0.5 * log_var) == e^(log σ) = σ.
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def show_image(x, batch_idx):
    x = x.view(batch_size, 28, 28) # Recover the shape of the image.

    fig = plt.figure()
    plt.imshow(x[batch_idx].cpu().numpy())
    

if __name__ == "__main__":
    dataset_path = '~/datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")


    batch_size = 100

    x_dim  = 784
    hidden_dim = 400
    latent_dim = 200

    lr = 1e-3

    epochs = 20
    
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
    
    
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    BCE_loss = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim) # Flatten pixel space shape=(1, 28, 28) to 1-axis shape=(784).
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finish!!")
    
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)


            break
        
    show_image(x, batch_idx=0)
    show_image(x_hat, batch_idx=0)
    
    
    
