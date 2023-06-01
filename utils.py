import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets

class AtariDataset(torch.utils.data.Dataset):
    def __init__(self, filename, resize=84):
        """
        Args:
            root_dir (string): Directory with all the images.
            position_data (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.d = pickle.load(open(filename,'rb'))
        self.resize = resize
        self.transform = transforms.Compose([
                # transforms.RandomSizedCrop(224),
                #transforms.Resize((resize,resize)),
                #transforms.RandomHorizontalFlip(),
                #transforms.Grayscale(1),
                transforms.ToTensor()
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # return len(os.listdir(self.root_dir)-1)
        return len(self.d['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # use only first frame
        image = np.squeeze(self.d['obs'][idx])[:,:,0]
        #resize(
        if self.transform:
            image = self.transform(image)

        return image, self.d['acs']

def plot_recon(opt, model, savefig=None):
    batch = model.saved_batch
    h = batch.shape[-1]
    w = batch.shape[-1]
    # 10x10 grid
    with torch.no_grad():
        x = model.forward(batch)['x_hat']
    n = np.sqrt(batch.shape[0]).astype(np.int32)
    I_batch = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_batch[i*h:(i+1)*h, j*w:(j+1)*w] = batch[i*n+j, :].reshape(h, w)

    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x[i*n+j, :].reshape(h, w)

    plt.figure(figsize=(16, 8))
    fig, axs = plt.subplots(1,2)
    axs[0].axis('off')
    axs[0].imshow(I_batch, cmap='gray_r')
    axs[1].axis('off')
    axs[1].imshow(I_generated, cmap='gray_r')
    if savefig is not None:
        plt.savefig(savefig)

def plot_disk(opt, model, h=84, w=84, savefig=None):
    n = 40
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)

    #r = np.logspace(0, 1, n)
    #y = np.linspace(0,2*pi,n)

    I_latent = np.empty((h*n, w*n))
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            z = torch.from_numpy(np.array([[xi, yi]]*opt.batch_size).astype(np.float32))
            x_hat = model.generate(z).detach().cpu().numpy()
            I_latent[(n-i-1)*h:(n-i)*w, j*h:(j+1)*w] = x_hat[20].reshape(h, w)
            
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(I_latent)#, cmap="gray_r")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()