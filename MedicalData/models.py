import torch
import torch.nn as nn
import torchvision
import numpy as np

class MedicalDataBase(nn.Module):
    def set_confounder(self, cfs, device):
        with torch.no_grad():
            if self.cf_dim == 1:
                self.cfs = nn.Parameter(cfs[:,None].to(device), requires_grad=False)
            elif self.cf_dim == 2:
                self.cfs = nn.Parameter(torch.cat((torch.ones(len(cfs), 1).to(device), torch.Tensor(cfs)[:,None]), dim=1).to(device), requires_grad=False)
            else:
                raise ValueError('cf_dim must be 1 or 2')
    
    def set_delta(self, trainloader, device):
        with torch.no_grad():
            XTX = torch.zeros((self.cf_dim, self.cf_dim)).to(device)
            XTH = torch.zeros((self.cf_dim, self.num_features)).to(device)
            for batch in trainloader:
                images = batch[0].to(device)
                cfs = batch[2].float().to(device)
                self.set_confounder(cfs, device)
                features = self.model(images)
                XTX += self.cfs.T @ self.cfs
                XTH += self.cfs.T @ features
                del images, cfs, features
                torch.cuda.empty_cache()
            self.delta = nn.Parameter(torch.inverse(XTX) @ XTH, requires_grad=False)
        
    def training_step(self, batch, device):
        images = batch[0].to(device) # Get images from the batch
        labels = batch[1].to(device) # Get labels from the batch
        if self.name == 'SSN':
            cfs = batch[2].float().to(device) # Get confounders from the batch
            self.set_confounder(cfs, device) # Set the confounders
        pred = self(images) # Get the model's predictions
        pred  = pred.squeeze() if self.num_classes == 1 else pred
        loss = self.criterion(pred, labels) # Calculate the loss
        return pred, loss # Return the output and loss
    
    def validation_step(self, batch, metric, device):
        with torch.no_grad():
            images = batch[0].to(device) # Get images from the batch
            labels = batch[1].to(device) # Get labels from the batch
            if self.name == 'SSN':
                cfs = batch[2].float().to(device) # Get confounders from the batch
                self.set_confounder(cfs, device)
            output = self(images) # Get the model's predictions
            output  = output.squeeze() if self.num_classes == 1 else output
            loss = self.criterion(output, labels) # Calculate the loss
            if self.num_classes > 1:
                acc = (output.argmax(dim=1) == labels).float().mean() # Calculate the accuracy
            else: 
                acc = (output.round() == labels).float().mean() # Calculate the accuracy
            auc = metric(output, labels) # Calculate the AUC
        return {'val_loss': loss.detach(), 'val_acc': acc, 'val_auc': auc} # Return the loss, accuracy and AUC
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs] # Get the losses
        batch_accs = [x['val_acc'] for x in outputs] # Get the accuracies
        batch_aucs = [x['val_auc'] for x in outputs] # Get the AUCs
        epoch_loss = torch.stack(batch_losses).mean() # Calculate the mean loss
        epoch_acc = torch.stack(batch_accs).mean() # Calculate the mean accuracy
        epoch_auc = torch.stack(batch_aucs).mean() # Calculate the mean AUC
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_auc': epoch_auc.item()} # Return the mean loss and accuracy
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, AUC: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['val_auc']))
        
class Baseline(MedicalDataBase): # Create the ResNet50 class
    def __init__(self, num_classes): # Initialize the class
        super(Baseline, self).__init__() # Initialize the super class
        self.name = 'Baseline' # Model name
        self.num_classes = num_classes # Number of classes
        self.criterion = nn.NLLLoss()
        # Load pretrained ResNet50 model
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # Add a new final fully connected layer with n_features output nodes classification
        in_features = self.model.fc.in_features # Get the number of input features
        self.model.fc = nn.Sequential( # Replace the fully connected layer
            nn.Linear(in_features, num_classes) # Add a new fully connected layer
        ) # End of the sequential
        if self.num_classes > 1:
            self.activation = nn.LogSoftmax(dim=1)
        else:
            self.activation = nn.LogSigmoid()
        
    def forward(self, x): # Forward propagation
        x = self.model(x) # Forward propagate through the model
        x = self.activation(x) # Apply the activation function
        return x

class SemiStructuredNet(MedicalDataBase):
    def __init__(self, batch_size, cf_dim, num_classes=4, num_features=128):
        super(SemiStructuredNet, self).__init__()
        self.name = 'SSN' # Model name
        self.cf_dim = cf_dim # Dimension of orthogonalisation
        self.num_features = num_features # Number of features in the final layer
        self.num_classes = num_classes
        self.criterion = nn.NLLLoss()
        self.cfs = nn.Parameter(torch.randn((batch_size, cf_dim)), requires_grad=False)
        self.delta = nn.Parameter(torch.randn((cf_dim, num_features)), requires_grad=False)
        self.orthogonalize_marker = True
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_features)
        )
        self.linear_predictor = nn.Linear(num_features, num_classes)
        if self.num_classes > 1:
            self.activation = nn.LogSoftmax(dim=1)
        else:
            self.activation = nn.LogSigmoid()
    
    def othogonalize(self, x):
        z = self.cfs
        if self.training:
            x = x - z @ torch.inverse(z.T @ z) @ z.T @ x
        else:
            x = x - z @ self.delta
        return x
    
    def forward(self, x):
        x = self.model(x)
        if self.orthogonalize_marker:
            x = self.othogonalize(x)
        x = self.linear_predictor(x)
        x = self.activation(x)
        return x