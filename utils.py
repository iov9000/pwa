import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
import torchvision.utils as vutils

from rietools_pt import np_exp_map

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

def plot_recon(opt, model, bw=False, savefig=None):
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

    if bw:
        cmap='gray_r'
    else:
        cmap=None
    plt.figure(figsize=(16, 8))
    fig, axs = plt.subplots(1,2)
    axs[0].axis('off')
    axs[0].imshow(I_batch, cmap=cmap)
    axs[1].axis('off')
    axs[1].imshow(I_generated, cmap=cmap)
    if savefig is not None:
        plt.savefig(savefig)

def plot_disk(opt, model, h=84, w=84, bw=False, savefig=None):
    n = 20
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)

    #r = np.logspace(0, 1, n)
    #y = np.linspace(0,2*pi,n)

    I_latent = np.empty((h*n, w*n))
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            if opt.vae_model == 'pwa':
                z = torch.from_numpy(np_exp_map(np.array([[xi, yi]]*opt.batch_size)).astype(np.float32))
            else:
                z = torch.from_numpy((np.array([[xi, yi]]*opt.batch_size)).astype(np.float32))

            x_hat = model.generate(z).detach().cpu().numpy()
            I_latent[(n-i-1)*h:(n-i)*w, j*h:(j+1)*w] = x_hat[20].reshape(h, w)
            
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    if bw:
        plt.imshow(I_latent, cmap="gray_r")
    else:
        plt.imshow(I_latent)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def plot_direction(opt, model, h=84, w=84, bw=False, savefig=None):
    n = 20
    # sample random direction 
    if opt.dim_z == 2:
        theta = np.random.uniform(0,2*np.pi)
        s0 = np.array([[np.cos(theta), np.sin(theta)]]).astype(np.float32)
    else:
        theta = np.expand_dims(np.random.uniform(0,2*np.pi, opt.dim_z).astype(np.float32),0)
        s0 = theta
    step = 1/n
    samples = []
    s = np.zeros_like(s0)
    for i in range(n):
        s += step*s0
        if opt.vae_model == 'pwa':
            z = torch.from_numpy(np_exp_map(s))
        else:
            z = torch.from_numpy(s)
        samples.append(model.generate(z).squeeze(0))

    samples = torch.stack(samples)
    vutils.save_image(samples.data, savefig, normalize=False, ncol=1, nrow=n)

### FASTA utils

protein_alphabet = "ACDEFGHIKLMNPQRSTVWYXBZJ"

def one_hot_encode(sequence, alphabet=protein_alphabet):
    encoding = np.zeros((len(sequence), len(alphabet)))
    for i, char in enumerate(sequence):
        encoding[i, alphabet.index(char)] = 1
    return encoding

def one_hot_decode(encoding, alphabet=protein_alphabet):
    sequence = ""
    for i in range(encoding.shape[0]):
        sequence += alphabet[np.argmax(encoding[i])]
    return sequence

class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file):
        self.fasta_file = fasta_file
        self.sequences = self.load_sequences()

    def load_sequences(self):
        with open(self.fasta_file, 'r') as file:
            sequences = {}
            current_sequence = ""
            current_header = ""
            
            for line in file:
                line = line.rstrip()  # Remove newline character
                
                if line.startswith(">"):
                    if current_header:
                        # Save the previous sequence if one exists
                        sequences[current_header] = one_hot_encode(current_sequence)
                    
                    # Start a new sequence
                    current_header = line[1:]  # Remove the ">" character
                    current_sequence = ""
                else:
                    # Continue the current sequence
                    current_sequence += line
            
            if current_header:
                # Save the last sequence in the file
                sequences[current_header] = one_hot_encode(current_sequence)
            
            return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = list(self.sequences.values())[idx]
        return torch.tensor(sequence, dtype=torch.float32)
