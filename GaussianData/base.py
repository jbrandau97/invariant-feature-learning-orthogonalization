import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metadatanorm import MetadataNorm
from utils import binary_acc

# Declare base class and models
class GaussianDataBase(nn.Module):
    def set_confounder(self, batch, model_type, cfs, cf_dim, device):
        with torch.no_grad():
            if model_type == 'SSN':
                if cf_dim == 1:
                    self.cfs = nn.Parameter(torch.Tensor(cfs)[:,None].to(device), requires_grad=False)
                elif cf_dim == 2:
                    self.cfs = nn.Parameter(torch.cat((torch.ones(len(cfs), 1), torch.Tensor(cfs)[:,None]), dim=1).to(device), requires_grad=False)
                else:
                    raise ValueError('cf_dim must be 1 or 2')
            elif model_type in ['Linear', 'Conv']:
                X_batch = np.zeros((len(batch), 3))
                X_batch[:,0] = batch['image'].float().cpu().detach().numpy()
                X_batch[:,1] = cfs.cpu().detach().numpy()
                X_batch[:,2] = np.ones((len(batch),))
                with torch.no_grad():
                    self.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)
            
    def set_delta(self, trainloader, cf_dim, num_features, device):
        with torch.no_grad():
            XTX = torch.zeros((cf_dim, cf_dim), requires_grad=False).to(device)
            XTH = torch.zeros((cf_dim, num_features), requires_grad=False).to(device)
            for batch in trainloader:
                data = batch['image'].float()
                target = batch['label'].float()
                cf_batch = batch['cfs'].float()
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    self.set_confounder(batch, cf_batch, cf_dim, device, model_type='SSN')
                features = self.fc2(F.relu(self.fc1(self.model_conv(data).view(-1, 18432))))
                XTX += self.cfs.T @ self.cfs
                XTH += self.cfs.T @ features
            self.delta = nn.Parameter(torch.inverse(XTX) @ XTH, requires_grad=False)
        
    def training_step(self, batch, model_type, cf_dim, device):
        data = batch['image'].float()
        target = batch['label'].float()
        cf_batch = batch['cfs'].float()
        data, target = data.to(device), target.to(device)
        if not (model_type == 'Baseline'):
            self.set_confounder(batch, model_type, cf_batch, cf_dim, device)
        y_pred, fc = self(data)
        loss = torch.nn.BCELoss(y_pred, target.unsqueeze(1))
        acc = binary_acc(y_pred, target.unsqueeze(1))
        return y_pred, fc, loss, acc
    
    def validation_step(self, batch, model_type, cf_dim, device):
        data = batch['image'].float()
        target = batch['label'].float()
        cf_batch = batch['cfs'].float()
        data, target = data.to(device), target.to(device)
        if not (model_type == 'Baseline'):
            self.set_confounder(batch, model_type, cf_batch, cf_dim, device)
        with torch.no_grad():
            y_pred, fc = self(data)
            loss = torch.nn.BCELoss(y_pred, target.unsqueeze(1))
            acc = binary_acc(y_pred, target.unsqueeze(1))
        return y_pred, fc, loss, acc
    
    #def validation_epoch_end(self, outputs):
    
class BaselineNet(GaussianDataBase):
    def __init__(self):
        """ Baseline CNN model with 2 convolutional layers and 2 linear layers. """
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(18432, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, output_fc=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        if output_fc:
            return x, fc
        else:
            return x

class MDN_Linear(GaussianDataBase):
    def __init__(self, dataset_size, batch_size, kernel):
        """ MDN-Linear model: Baseline CNN model with 2 convolutional and 2 linear layers with MDN applied
            to the last linear layer before the output layer. 
        Args:
          dataset_size (int): size of dataset
          batch_size (int): batch size
          kernel (2d vector): precalculated kernel for MDN based on the vector X of confounders (X^TX)^-1.
              kernel needs to be set before training, and cfs needs to be set during training for each batch.
        """
        super(MDN_Linear, self).__init__()
        self.N = batch_size
        self.C = kernel.shape[0] 
        self.kernel = kernel
        self.cfs = nn.Parameter(torch.randn(batch_size, self.C), requires_grad=False)
        self.dataset_size = dataset_size
         
        # Convolutional and MDN layers
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(18432, 84)
        self.metadatanorm = MetadataNorm(self.N, self.kernel, self.dataset_size, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, output_fc=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        self.metadatanorm.cfs = self.cfs
        x = self.metadatanorm(x)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        if output_fc:
            return x, fc
        else:
            return x
    
class MDN_Conv(GaussianDataBase):
    def __init__(self, dataset_size, batch_size, kernel):
        """ MDN-Conv model: Baseline CNN model with 2 convolutional and 2 linear layers with MDN applied
            to every convolutional layer and the last linear layer before the output layer.
        Args:
          dataset_size (int): size of dataset
          batch_size (int): batch size
          kernel (2d vector): precalculated kernel for MDN based on the vector X of confounders (X^TX)^-1.
              kernel needs to be set before training, and cfs needs to be set during training for each batch.
        """
        super(MDN_Conv, self).__init__()
        self.N = batch_size
        self.C = kernel.shape[0] 
        self.kernel = kernel
        self.cfs = nn.Parameter(torch.randn(batch_size, self.C), requires_grad=False)
        self.dataset_size = dataset_size
 
        # Convolutional and MDN layers
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.metadatanorm1 = MetadataNorm(self.N, self.kernel, self.dataset_size, 16*28*28)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.metadatanorm2 = MetadataNorm(self.N, self.kernel, self.dataset_size, 32*24*24)
        self.fc1 = nn.Linear(18432, 84)
        self.metadatanorm3 = MetadataNorm(self.N, self.kernel, self.dataset_size, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, output_fc=False):
        x = self.conv1(x)
        self.metadatanorm1.cfs = self.cfs
        x = self.metadatanorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        self.metadatanorm2.cfs = self.cfs
        x = self.metadatanorm2(x) 
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        self.metadatanorm3.cfs = self.cfs
        x = self.metadatanorm3(x)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        if output_fc:
            return x, fc
        else:
            return x
    
class SSN(GaussianDataBase):
    def __init__(self, batch_size, cf_dim, num_features):
        super(SSN, self).__init__()
        self.cfs = nn.Parameter(torch.randn((batch_size, cf_dim)), requires_grad=False)
        self.delta = nn.Parameter(torch.randn((cf_dim, num_features)), requires_grad=False)
        self.orthogonalize_marker = True
        self.model_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU())
        self.fc1 = nn.Linear(18432, 84)
        self.fc2 = nn.Linear(84, num_features)
        self.linear_predictor = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def orthogonalize(self, x):  
        z = self.cfs
        if self.training: 
            x = x - z @ torch.inverse(z.T @ z) @ z.T @ x
        else:
            x = x - z @ self.delta
        return x  
        
    def forward(self, data, output_fc=False):
        x = self.model_conv(data)
        x = x.view(-1, 18432)
        x = self.fc1(x)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.orthogonalize(x)
        x = self.linear_predictor(x)
        x = self.sigmoid(x)
        if output_fc:
            return x, fc
        else:
            return x