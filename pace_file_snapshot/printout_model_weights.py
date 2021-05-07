# Official training implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import sys
import pickle
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# Parameters for training
batch_size = 64
test_batch_size = 1000
epochs = 14
lr = 1.0
gamma = 0.7
no_cuda = False
dry_run = False
seed = 1
log_interval = 10
save_model = False

use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=False,#True,
                   transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                   transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

import copy

# Implement client logic
class ClientNet():
    def __init__(self, dataset, B=16, E=1, optimizerlr=1, schedgamma=0.7):
        self.model = Net().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=optimizerlr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=schedgamma)
        self.loss_history = []
        self.B = B
        self.E = E
        train_kwargs = {'batch_size': B}
        self.train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)
        
    def update(self, dry_run=False, max_n=np.inf, weights=None):
        if weights is not None:
            self.model.load_state_dict(weights)
        self.model.train()
        for e in range(self.E):
            self.loss_history.append([])
            # How to only load a limited amount of data?
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                self.loss_history[-1].append(loss.item())

                if dry_run or batch_idx*self.B >= max_n:
                    break
#         print(len(self.loss_history[-1]))
        
# Define some parameters for DFL
K = 10 # Number of clients
C = 1
B = 64#10 # Batch size in clients
E = 1 # Epochs per client update (>= 1)
n = len(dataset1) # Number of training samples in total
num_rounds = 500#200
EE = 1/20 # (1/num_rounds) # Portion of data per epoch (<= 1)
max_n = n/K * EE

import utils
S = 1.0
Mean = 1
V = 0.7

# script_filiename = 'MNIST_DFL_diverseClient_dataset_K{}_C{}_E{}_S{}_M{}_V{}_'.format(K,C,E*EE,S,Mean,V)
script_filiename = "MNIST_DFL_full_dataset_K10_C1_E10_"
script_filename = 'tmp_data/'+script_filiename
file_identifier = ''
if len(sys.argv) > 1:
    file_identifier = sys.argv[1]

# Retrieve agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(test_loader, B, E=E, optimizerlr=lr))
try:
    for i,cl in enumerate(clients_DFL):
        cl.model.load_state_dict(torch.load(script_filename+file_identifier+'_client{}'.format(i)))
        with open(script_filename+file_identifier+'_client{}_loss_history'.format(i), "rb") as fp:
            cl.loss_history = pickle.load(fp)
    dists = np.load(script_filename+file_identifier+'_dists_history_temp.npy')
except:
    print("Did not find model weights that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")

#for i in range(K):
#    print("Model {}'s weights:".format(i))
#    print(clients_DFL[i].model.state_dict())

for key in clients_DFL[0].model.state_dict().keys():
    weight_sum = clients_DFL[0].model.state_dict()[key] * 0
    for i in range(K):
        print("Model {}'s weights for key {}:".format(i, key))
        print(clients_DFL[i].model.state_dict()[key])
        weight_sum += clients_DFL[i].model.state_dict()[key]
    print("Total average weight for key {} is:".format(key))
    print(weight_sum/K)

