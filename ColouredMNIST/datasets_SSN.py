import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class ColoredMNIST(Dataset):
    def __init__(self, images, labels, colors, true_labels):
        self.images = images
        self.labels = labels
        self.colors = colors
        self.true_labels = true_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        colors = self.colors[idx]
        true_labels = self.true_labels[idx]
        return {
            'images': images,
            'labels': labels,
            'colors': colors,
            'true_labels': true_labels
        }
        
def make_environment(images, labels, e, downsampling_factor=2, flip_labels=True):
  # Source: https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    images = images.reshape((-1, 28, 28))[:, ::downsampling_factor, ::downsampling_factor] # 2x subsample for computational convenience
    # Assign a binary label based on the digit; flip label with probability 0.25
    true_labels = (labels > 4).float() # Assigns binary label of 0 for 0-4 and 1 for 5-9
    labels = torch_xor(true_labels, torch_bernoulli(0.25, len(true_labels))) if flip_labels else true_labels # Flips the label with probability 0.25, flips if Bernoulli is 1
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels))) # Flips the label with probability e
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1) # Adds color channel dimension
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0 
    # Alternative: create third color channel with zeros to get RGB images
    #images[torch.tensor(range(len(images))), 2, :, :] *= 0 # Zero out blue channel
    # Alternative: color background instead of foreground
    #images[images == 0] += 255 # Set background
    #images[torch.tensor(range(len(images))), 2, :, :] *= 0 # Zero out blue channel
    #images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'colors': colors[:, None], # Added color vector needed for orthogonalization
      'true_labels': true_labels[:, None] # Added true label vector for calculation of correlations
    }
        
def batch_mean_and_sd(dataloader):
    # Source: https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in dataloader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std
        
def prepare_data():
    # Loading and preparing the data
    # Specify transforms to apply to the data
    transform = transforms.Compose([transforms.ToTensor(),]) # Convert image to tensor

    # Load the data and trasnform to Tensor
    trainset = datasets.MNIST('data/MNIST_classification/trainset', download=True, train=True, transform=transform)
    testset = datasets.MNIST('data/MNIST_classification/testset', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=8)

    # Compute mean and std of trainset and valset
    mean_train, std_train = batch_mean_and_sd(trainloader)
    mean_test, std_test = batch_mean_and_sd(testloader)
    print(f"Mean and std of trainset before normalization: {mean_train}, {std_train}")
    print(f"Mean and std of valset before normalization: {mean_test}, {std_test}")

    # Define transforms with normalization
    transform_normalize_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_train[0].item(), std_train[0].item()),])
    transform_normalize_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_test[0].item(), std_test[0].item()),])

    # Normalize the data
    trainset = datasets.MNIST('data/MNIST_classification/trainset', download=True, train=True, transform=transform_normalize_train)
    testset = datasets.MNIST('data/MNIST_classification/testset', download=True, train=False, transform=transform_normalize_val)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=8)

    # Check the mean and std of the normalized data
    mean_train, std_train = batch_mean_and_sd(trainloader)
    mean_test, std_test = batch_mean_and_sd(testloader)
    print(f"Mean and std of trainset after normalization: {mean_train}, {std_train}")
    print(f"Mean and std of valset after normalization: {mean_test}, {std_test}")
    
    return {'trainset': trainset, 'testset': testset}
    
def create_dataloader(trainset, testset, sample_size=[50000, 10000], train_val_split=0.8, p_train=0.1, p_test=0.9, downsampling_factor=2, flip_labels=True, batch_size=100):
    # Create ColoredMNIST dataset and plot first 10 images of each environment
    mnist_train = (trainset.data[:sample_size[0]], trainset.targets[:sample_size[0]]) # (images, labels)
    mnist_test = (testset.data[:sample_size[1]], testset.targets[:sample_size[1]]) # (images, labels)
    train_val_split = int(train_val_split*sample_size[0]) # Split the training set into training and validation set

    # Create environments with flipped labels
    envs = [
        make_environment(mnist_train[0], mnist_train[1], p_train, downsampling_factor, flip_labels), # Environment for training and validation with 20% of the colors flipped
        make_environment(mnist_test[0], mnist_test[1], p_test, downsampling_factor, flip_labels) # Environment for testing with 90% of the colors flipped
    ]

    traindata = ColoredMNIST(envs[0]['images'][:train_val_split], 
                            envs[0]['labels'][:train_val_split], 
                            envs[0]['colors'][:train_val_split],
                            envs[0]['true_labels'][:train_val_split]) # Create ColoredMNIST dataset training set
    valdata = ColoredMNIST(envs[0]['images'][train_val_split:], 
                        envs[0]['labels'][train_val_split:], 
                        envs[0]['colors'][train_val_split:],
                        envs[0]['true_labels'][train_val_split:]) # Create ColoredMNIST dataset validation set
    testdata = ColoredMNIST(envs[1]['images'], 
                            envs[1]['labels'], 
                            envs[1]['colors'],
                            envs[1]['true_labels']) # Create ColoredMNIST dataset test set
    trainloader = DataLoader(traindata, batch_size, shuffle=True) # Create DataLoader for training set
    valloader = DataLoader(valdata, batch_size, shuffle=True) # Create DataLoader for validation set
    testloader = DataLoader(testdata, batch_size, shuffle=True) # Create DataLoader for test set
    
    print("Dataloader created!")
    
    return {'train': trainloader, 'val': valloader, 'test': testloader}
  
def plot_colored_mnist(images, p):
  labels = images['labels']
  colors = images['colors']
  images = images['images']
  fig, axs = plt.subplots(3, 3, layout='compressed')
  #fig.suptitle(r'Training Environment with $p^{e}=$' + str(p))
  k = 0
  for i in range(3):
    for j in range(3):
      axs[j,i].set_title(f"Label: {int(labels[k].item())}", fontdict={'fontsize': 18})
      axs[j,i].set_xticks([])
      axs[j,i].set_yticks([])
      if colors[k][0] == 0:
        axs[j,i].imshow(images[k][0], cmap="Blues")
      else:
        axs[j,i].imshow(images[k][1], cmap="Reds")
      k += 1
  plt.savefig(f"plots/ColoredMNIST/environment_{p}.pdf")