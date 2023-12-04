import torch 
import torchvision
import numpy as np
import pandas as pd
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
from pathlib import Path
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from models import Baseline, SemiStructuredNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # GPU
downloads_path = str(Path.home() / "Downloads") # Downloads folder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:734"

def evaluate(model, valloader, device):
    model.eval()
    if model.num_classes > 1:
        metric = MulticlassAUROC(num_classes=model.num_classes, average='macro', thresholds=None) # Initialize AUROC metric
    else: 
        metric = BinaryAUROC() # Initialize AUROC metric
    outputs = [model.validation_step(batch, metric, device) for batch in valloader]
    return model.validation_epoch_end(outputs)

def fit(model, optimizer, scheduler, trainloader, valloader, epochs, device, print_results=True):
    torch.cuda.empty_cache() # Empty GPU cache
    train_history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in trainloader:
            optimizer.zero_grad()
            _, loss = model.training_step(batch, device)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if model.name == 'SSN':
                model.set_delta(trainloader, device)
            del batch
            torch.cuda.empty_cache()
        model.eval()
        result = evaluate(model, valloader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        if print_results:
            model.epoch_end(epoch, result)
        train_history.append(result)
    return train_history

# Check if data objects already exist in directory and load them, if not, create them
if not (os.path.exists("control_p.pt") and os.path.exists("disease_p.pt") and os.path.exists("control_a.pt") and os.path.exists("disease_a.pt")):
    # Loading the data for the pediatric dataset and resizing the images to 200x200
    dataset_pediatric = ImageFolder(root=downloads_path + "/archive/chest_xray/chest_xray/train", 
                                    transform=transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()]))

    # Loading the data for the adult dataset and resizing the images to 200x200
    dataset_adult = ImageFolder(root=downloads_path + "/rsna-pneumonia-detection-challenge/train",
                                transform=transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()]))

    # Check numbers of disease and control in each age group to identify the minority class
    disease_pediatric = dataset_pediatric.targets.count(1) # Count the number of disease images  
    control_pediatric = dataset_pediatric.targets.count(0) # Count the number of control images
    disease_adult = dataset_adult.targets.count(1) # Count the number of disease images
    control_adult = dataset_adult.targets.count(0) # Count the number of control images
    print(f'Pediatric: Control: {control_pediatric}, Disease: {disease_pediatric}\nAdult: Control: {control_adult}, Disease: {disease_adult}') # Print the numbers

    # Set minority class to size 1000 and sample down the majority class to match
    minority_size = 1000 # Set the number of control images to 1000
    
    # Remove control images and targets from the pediatric dataset to sample down to match number of control images of minority class
    for i in range(control_pediatric - minority_size): # Iterate over the number of disease images minus the number of control images
        dataset_pediatric.imgs.remove(dataset_pediatric.imgs[0]) # Remove the last image
        dataset_pediatric.targets.remove(dataset_pediatric.targets[0]) # Remove the last target
    
    # Remove disease images and targets from the pediatric dataset to sample down to match number of control images of minority class
    for i in range(disease_pediatric - minority_size): # Iterate over the number of disease images minus the number of control images
        dataset_pediatric.imgs.remove(dataset_pediatric.imgs[-1]) # Remove the last image
        dataset_pediatric.targets.remove(dataset_pediatric.targets[-1]) # Remove the last target

    # Remove control images and targets from the adult dataset to sample down to match number of control images of minority class
    for i in range(control_adult - minority_size): # Iterate over the number of control images minus the number of control images
        dataset_adult.imgs.remove(dataset_adult.imgs[0]) # Remove the first image
        dataset_adult.targets.remove(dataset_adult.targets[0]) # Remove the first target
        
    # Remove disease images and targets from the adult dataset to sample down to match number of control images of minority class
    for i in range(disease_adult - minority_size): # Iterate over the number of disease images minus the number of control images
        dataset_adult.imgs.remove(dataset_adult.imgs[-1]) # Remove the last image
        dataset_adult.targets.remove(dataset_adult.targets[-1]) # Remove the last target
        
    # Check numbers of disease and control in each age group to see if they match
    disease_pediatric = dataset_pediatric.targets.count(1) # Count the number of disease images
    control_pediatric = dataset_pediatric.targets.count(0) # Count the number of control images
    disease_adult = dataset_adult.targets.count(1) # Count the number of disease images
    control_adult = dataset_adult.targets.count(0) # Count the number of control images
    print(f'Pediatric: Control: {control_pediatric}, Disease: {disease_pediatric}\nAdult: Control: {control_adult}, Disease: {disease_adult}') # Print the numbers

    # Split the pediatric and adult datasets into control and disease yielding 4 datasets for the different classes
    control_p, disease_p, control_a, disease_a = [], [], [], [] # Initialize the lists
    for i in range(len(dataset_pediatric)): # Iterate over the pediatric dataset
        if dataset_pediatric[i][1] == 0: # If the image is control
            control_p.append(dataset_pediatric[i]) # Append the image to the control list
        else: # If the image is disease
            disease_p.append(dataset_pediatric[i]) # Append the image to the disease list
    for i in range(len(dataset_adult)): # Iterate over the adult dataset
        if dataset_adult[i][1] == 0: # If the image is control
            control_a.append(dataset_adult[i]) # Append the image to the control list
        else: # If the image is disease
            disease_a.append(dataset_adult[i]) # Append the image to the disease list
    
    '''        
    # Rename classes; 0: control pediatric, 1: disease pediatric, 2: control adult, 3: disease adult
    for i in range(len(control_a)): # Iterate over the control adult dataset
        control_a[i] = (control_a[i][0], 2) # Rename the class
    for i in range(len(disease_a)): # Iterate over the disease adult dataset
        disease_a[i] = (disease_a[i][0], 3) # Rename the class
    '''
    
    # Save the datasets
    torch.save(control_p, "control_p.pt") # Save the control pediatric dataset
    torch.save(disease_p, "disease_p.pt") # Save the disease pediatric dataset
    torch.save(control_a, "control_a.pt") # Save the control adult dataset
    torch.save(disease_a, "disease_a.pt") # Save the disease adult dataset
else:
    # Load the datasets
    control_p = torch.load("control_p.pt") # Load the control pediatric dataset
    disease_p = torch.load("disease_p.pt") # Load the disease pediatric dataset
    control_a = torch.load("control_a.pt") # Load the control adult dataset
    disease_a = torch.load("disease_a.pt") # Load the disease adult dataset

# Add confounder label to the datasets
for i in range(len(control_p)): # Iterate over the control pediatric dataset
    control_p[i] = control_p[i] + (0,) # Add the confounder label
for i in range(len(disease_p)): # Iterate over the disease pediatric dataset
    disease_p[i] = disease_p[i] + (0,) # Add the confounder label
for i in range(len(control_a)): # Iterate over the control adult dataset
    control_a[i] = control_a[i] + (1,) # Add the confounder label
for i in range(len(disease_a)): # Iterate over the disease adult dataset
    disease_a[i] = disease_a[i] + (1,) # Add the confounder label
    
# Create the test dataset by sampling 10% of the images from each class
testdata_size = 100 #int(0.1 * len(control_p)) # Calculate the number of images to sample
sampling_indices = np.random.choice(range(1000), testdata_size, replace=False) # Sample the indices
testdata = [] # Initialize the list
for i in range(testdata_size): # Iterate over the number of images to sample
    testdata.append(control_p[sampling_indices[i]]) # Append the image to the test dataset
    testdata.append(disease_p[sampling_indices[i]]) # Append the image to the test dataset
    testdata.append(control_a[sampling_indices[i]]) # Append the image to the test dataset
    testdata.append(disease_a[sampling_indices[i]]) # Append the image to the test dataset
for i in sorted(sampling_indices, reverse=True): # Iterate over the indices in reverse order
    del control_p[i] # Delete the image from the control list
    del control_a[i] # Delete the image from the control list
    del disease_p[i] # Delete the image from the disease list
    del disease_a[i] # Delete the image from the disease list
    
# Create the balanced training dataset by sampling the same number of images from each class
length = 500 #int(len(control_p)/2) # Calculate the number of images to sample
sampling_indices = np.random.choice(range(900), length, replace=False) #np.random.choice(range(len(control_p)), length, replace=False) # Sample the indices
balanced = [] # Initialize the list
for i in range(length): # Iterate over the number of images to sample
    balanced.append(control_p[sampling_indices[i]]) # Append the image to the balanced dataset
    balanced.append(disease_p[sampling_indices[i]]) # Append the image to the balanced dataset
    balanced.append(control_a[sampling_indices[i]]) # Append the image to the balanced dataset
    balanced.append(disease_a[sampling_indices[i]]) # Append the image to the balanced dataset
    
# Create confounder datasets for total confounding
# Total confounding: Create traindata from half pediatric disease and half adult control or vice versa
# 1) Pediatric disease data and adult control data
# 2) Pediatric control data and adult disease data

length_total = 500 #len(control_p) # Length of half the training dataset
sampling_indices = np.random.choice(range(900), length_total, replace=False) # Sample the indices
total_1 = [] # Initialize the list
for i in range(length_total): # Iterate over the number of images to sample
    total_1.append(disease_p[sampling_indices[i]]) # Append the image to the total dataset
    total_1.append(control_a[sampling_indices[i]]) # Append the image to the total dataset
sampling_indices = np.random.choice(range(900), length_total, replace=False) # Sample the indices
total_2 = [] # Initialize the list
for i in range(length_total): # Iterate over the number of images to sample
    total_2.append(disease_a[sampling_indices[i]]) # Append the image to the total dataset
    total_2.append(control_p[sampling_indices[i]]) # Append the image to the total dataset
    
# Generate light confounding datasets in steps of 5 percentage points
light_confounded_data = {} # Initialize the dictionary
for step in range(1,10):
    length = int(500 - 50*step) # Calculate the number of images to sample
    sampling_indices_1 = np.random.choice(range(900), length, replace=False) # Sample the indices
    sampling_indices_2 = np.random.choice(range(900), 500-length, replace=False) # Sample the indices
    light_confounded_data[f'{int(50-5*step)}-{int(5*step)}'] = [] # Initialize the list
    for i in range(length):
        light_confounded_data[f'{int(50-5*step)}-{int(5*step)}'].append(disease_p[sampling_indices_1[i]])
        light_confounded_data[f'{int(50-5*step)}-{int(5*step)}'].append(control_a[sampling_indices_1[i]])
    for j in range(500-length):
        light_confounded_data[f'{int(50-5*step)}-{int(5*step)}'].append(disease_a[sampling_indices_2[j]])
        light_confounded_data[f'{int(50-5*step)}-{int(5*step)}'].append(control_p[sampling_indices_2[j]])
    
# Split traindata in training and validation (internal testing) data
total_1_train, total_1_val = torch.utils.data.random_split(total_1, [800, 200]) #[int(0.8*len(total_1)) + 1 if (len(total_1) % 2) == 0 else int(0.8*len(total_1)), int(0.2*len(total_1))]) # Split the dataset
total_2_train, total_2_val = torch.utils.data.random_split(total_2, [800, 200]) #[int(0.8*len(total_2)) + 1 if (len(total_2) % 2) == 0 else int(0.8*len(total_2)), int(0.2*len(total_2))]) # Split the dataset
balanced_train, balanced_val = torch.utils.data.random_split(balanced, [1600, 400]) #[int(0.8*len(balanced)) + 1 if (len(balanced) % 2) == 0 else int(0.8*len(balanced)), int(0.2*len(balanced))]) # Split the dataset
light_confounded_train_val = {} # Initialize the dictionary
for key in light_confounded_data.keys():
    light_confounded_train_val[key + '_train'], light_confounded_train_val[key + '_val'] = torch.utils.data.random_split(light_confounded_data[key], [800, 200]) # Split the dataset
torch.save(light_confounded_data, "light_confounded_data.pt") # Save the light confounded datasets
torch.save(light_confounded_train_val, "light_confounded_train_val.pt") # Save the light confounded training and validation datasets

train_results = {'Baseline': {}, 'SSN_1': {}, 'SSN_2': {}} # Initialize the dictionary
test_balanced_results = {'Baseline': {}, 'SSN_1': {}, 'SSN_2': {}} # Initialize the dictionary
test_inverse_results = {'Baseline': {}, 'SSN_1': {}, 'SSN_2': {}} # Initialize the dictionary
keys = [f'{int(50-5*step)}-{int(5*step)}' for step in range(1,10)] # Initialize the keys
for key in keys:
    print(f'Light confounding: {key}')
    train_results['Baseline'][key] = {} # Initialize the dictionary
    train_results['SSN_1'][key] = {} # Initialize the dictionary
    train_results['SSN_2'][key] = {} # Initialize the dictionary
    test_balanced_results['Baseline'][key] = {} # Initialize the dictionary
    test_balanced_results['SSN_1'][key] = {} # Initialize the dictionary
    test_balanced_results['SSN_2'][key] = {} # Initialize the dictionary
    test_inverse_results['Baseline'][key] = {} # Initialize the dictionary
    test_inverse_results['SSN_1'][key] = {} # Initialize the dictionary
    test_inverse_results['SSN_2'][key] = {} # Initialize the dictionary
    for run in range(10):
        print(f'Run: {run+1}/10')
        trainloader = DataLoader(light_confounded_train_val[key + '_train'], batch_size=50, shuffle=True) # Get the trainloader
        valloader = DataLoader(light_confounded_train_val[key + '_val'], batch_size=50, shuffle=True) # Get the valloader
        Model_Baseline = Baseline(num_classes=2).to(device)
        optimizer_Baseline = optim.Adam(Model_Baseline.parameters(), lr=0.001, weight_decay=0.0001) # Create optimizer
        scheduler_Baseline = torch.optim.lr_scheduler.OneCycleLR(optimizer_Baseline, max_lr=0.01, epochs=15, steps_per_epoch=len(trainloader))
        Model_SSN_1 = SemiStructuredNet(batch_size=50, cf_dim=1, num_classes=2, num_features=128).to(device)
        optimizer_SSN_1 = optim.Adam(Model_SSN_1.parameters(), lr=0.001, weight_decay=0.0001) # Create optimizer
        scheduler_SSN_1 = torch.optim.lr_scheduler.OneCycleLR(optimizer_SSN_1, max_lr=0.01, epochs=15, steps_per_epoch=len(trainloader))
        Model_SSN_2 = SemiStructuredNet(batch_size=50, cf_dim=2, num_classes=2, num_features=128).to(device)
        optimizer_SSN_2 = optim.Adam(Model_SSN_2.parameters(), lr=0.001, weight_decay=0.0001) # Create optimizer
        scheduler_SSN_2 = torch.optim.lr_scheduler.OneCycleLR(optimizer_SSN_2, max_lr=0.01, epochs=15, steps_per_epoch=len(trainloader))
        # Train the baseline model
        train_results['Baseline'][key][run] = fit(Model_Baseline, optimizer_Baseline, scheduler_Baseline, trainloader, valloader, epochs = 15, device=device, print_results=False)
        # Train the SSN models
        train_results['SSN_1'][key][run] = fit(Model_SSN_1, optimizer_SSN_1, scheduler_SSN_1, trainloader, valloader, epochs = 15, device=device, print_results=False)
        train_results['SSN_2'][key][run] = fit(Model_SSN_2, optimizer_SSN_2, scheduler_SSN_2, trainloader, valloader, epochs = 15, device=device, print_results=False)
        # Test the models
        testloader = DataLoader(testdata, batch_size=50, shuffle=True) # Get the testloader
        test_balanced_results['Baseline'][key][run] = evaluate(Model_Baseline, testloader, device)['val_auc']
        test_balanced_results['SSN_1'][key][run] = evaluate(Model_SSN_1, testloader, device)['val_auc']
        test_balanced_results['SSN_2'][key][run] = evaluate(Model_SSN_2, testloader, device)['val_auc']
        testloader = DataLoader(light_confounded_train_val[key.split('-')[1] + '-' + key.split('-')[0] + '_val'], batch_size=50, shuffle=True) # Get the testloader
        test_inverse_results['Baseline'][key][run] = evaluate(Model_Baseline, testloader, device)['val_auc']
        test_inverse_results['SSN_1'][key][run] = evaluate(Model_SSN_1, testloader, device)['val_auc']
        test_inverse_results['SSN_2'][key][run] = evaluate(Model_SSN_2, testloader, device)['val_auc']
        # Save fitted models
        torch.save(Model_Baseline.state_dict(), f"fitted_models/Baseline_{key}_{run}.pt")
        torch.save(Model_SSN_1.state_dict(), f"fitted_models/SSN_1_{key}_{run}.pt")
        torch.save(Model_SSN_2.state_dict(), f"fitted_models/SSN_2_{key}_{run}.pt")
torch.save(train_results, "train_results.pt") # Save the training results
torch.save(test_balanced_results, "test_balanced_results.pt") # Save the test results
torch.save(test_inverse_results, "test_inverse_results.pt") # Save the test results
print('Done!')