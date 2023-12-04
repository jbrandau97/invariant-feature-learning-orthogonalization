import torch
import numpy as np
from torch import nn

#########################
# Base classes for models
#########################

class ColoredMNIST(nn.Module):
    def set_confounders(self, cfs, cf_dim, device):
        with torch.no_grad():
            if cf_dim == 1:
                self.cfs = nn.Parameter(torch.Tensor(cfs).to(device), requires_grad=False)
            elif cf_dim == 2:
                self.cfs = nn.Parameter(torch.cat((torch.ones(len(cfs), 1).to(device), torch.Tensor(cfs)), dim=1).to(device), requires_grad=False)
            else:
                raise ValueError("cf_dim is out of the specified range")
    
    def set_delta(self, trainloader, cf_dim, num_features, device):
        with torch.no_grad():
            XTX = torch.zeros((cf_dim, cf_dim), requires_grad=False).to(device)
            XTH = torch.zeros((cf_dim, num_features), requires_grad=False).to(device)
            for batch in trainloader:
                images = batch['images'].to(device) 
                colors = batch['colors'].to(device)
                self.set_confounders(colors, cf_dim, device)
                images = images.view(images.shape[0], -1)
                features = self.model(images)
                XTX += self.cfs.T @ self.cfs
                XTH += self.cfs.T @ features
            self.delta = nn.Parameter(torch.inverse(XTX) @ XTH, requires_grad=False)
            
    def training_step(self, batch, model_type, cf_dim, device):
        images = batch['images'].to(device) 
        labels = batch['labels'].to(device)
        if model_type == 'SSN':
            colors = batch['colors'].to(device)
            self.set_confounders(colors, cf_dim, device)
        images = images.view(images.shape[0], -1)
        pred = self(images)
        loss = nn.functional.binary_cross_entropy(pred, labels)
        return pred, loss
    
    def validation_step(self, batch, model_type, cf_dim, device):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        if model_type == 'SSN':
            colors = batch['colors'].to(device)
            self.set_confounders(colors, cf_dim, device)
        images = images.view(images.shape[0], -1)
        out = self(images)
        loss = nn.functional.binary_cross_entropy(out, labels)
        acc = torch.sum(out.round() == labels) / float(len(labels))
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2%}".format(epoch, result['val_loss'], result['val_acc']))

#########################
# Models
#########################

class Baseline(ColoredMNIST):
    def __init__(self, num_channel=2, num_pixel=14, num_features=32):
        super(Baseline, self).__init__()
        lin1 = nn.Linear(num_channel * num_pixel**2, 390)
        lin2 = nn.Linear(390, 390)
        lin3 = nn.Linear(390, num_features)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.model = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        self.linear_predictor = nn.Linear(32, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        return self.activation(self.linear_predictor(self.model(x)))
    
class IRM(nn.Module):
    def __init__(self):
        super(IRM, self).__init__()
        lin1 = nn.Linear(2 * 14 * 14, 390)
        lin2 = nn.Linear(390, 390)
        lin3 = nn.Linear(390, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.model = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        
    def forward(self, input):
        out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out
    
class SemiStructuredNet(ColoredMNIST):
    def __init__(self, batch_size, cf_dim, num_features=32):
        super(SemiStructuredNet, self).__init__()
        self.cfs = nn.Parameter(torch.randn((batch_size, cf_dim)), requires_grad=False)
        self.delta = nn.Parameter(torch.randn((cf_dim, num_features)), requires_grad=False)
        self.orthogonalize_marker = True
        lin1 = nn.Linear(2 * 14 * 14, 390)
        lin2 = nn.Linear(390, 390)
        lin3 = nn.Linear(390, num_features)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.model = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        self.linear_predictor = nn.Linear(32, 1)
        self.activation = nn.Sigmoid()
      
    def orthogonalize(self, x):
        z = self.cfs
        if self.training: 
            x = x - z @ torch.inverse(z.T @ z) @ z.T @ x
        else:
            x = x - z @ self.delta
        return x
    
    def forward(self, x):
        out = self.model(x)
        if self.orthogonalize_marker:
            out = self.orthogonalize(out)
        out = self.linear_predictor(out)
        out = self.activation(out)
        return out