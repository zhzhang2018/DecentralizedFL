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
# Implement server logic
class ServerNet():
    def __init__(self):
        self.model = Net().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
    
    def update(self, clients):
        # Take averages of each client.
        # https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/235b8f0ab161825694ee73874295d773b0d3f23f/src/utils.py
        # https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008
        w_avg = copy.deepcopy(clients[0].model.state_dict())
        for key in w_avg.keys():
            for i in range(1, len(clients)):
                w_avg[key] += clients[i].model.state_dict()[key]
            w_avg[key] = torch.div(w_avg[key], len(clients))
        self.model.load_state_dict(w_avg)

# Implement client logic
class ClientNet():
    def __init__(self, dataset, B=16, E=1):
        self.model = Net().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
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
        
# Try to see if you can train distributedly - i.e. average between several separate models
# Implement train and test logic. This one doens't have a server, and I assume we can let them all update together.
# dists: expected to be a K x K x num_avg_iter x num_rounds array to store distances
def train_DFL(adj, clients, dists, num_rounds=1, dry_run=False, debug=False, CK=10, max_n=np.inf, test_loader=None,
              avg_weight=1, test_all_client=False, avg_alg=None,
              num_avg_iter=-1, max_num_avg_iter=1000, avg_error_thres=0.01,
              past_params=None, save_model_to='default_model_saving_name_', bother_saving=0):
    test_loss = []
    test_accuracy = []
    start_round_ind = 0
    if past_params is not None:
        test_loss = past_params[0]
        test_accuracy = past_params[1]
        start_round_ind = past_params[2]
    for e in range(start_round_ind,num_rounds):
        # Update each client
        for m in clients:
            m.update(dry_run, max_n=max_n)
        # Aggregation for each client. How it's done depends on the num_avg_iter argument:
        # num_avg_iter > 0: Execute at most this number of consensus averaging.
        # num_avg_iter < 0: Execute until the system reaches consensus (with pairwise errors smaller than avg_error_thres)
        print("Finished updating for {}-th round".format(e))
        reached_consensus = False
        for k in range(max_num_avg_iter):
            if reached_consensus:
                break
            if (k >= num_avg_iter and num_avg_iter > 0):
                break
            
            # Set up variables where we do the averaging calculation without disturbing the previous weights
            new_models = []
            for i in range(len(clients)):
                new_models.append( copy.deepcopy(clients[i].model.state_dict()) )

            reached_consensus = True
            # Start averaging towards the goal. Here we use equal neighbor averaging method.
            for i in range(len(clients)):
                for j in range(len(clients)):
                    # Record each key's value, while also keeping track of distance
                    client_wise_dist = 0
                    for key in new_models[j].keys():
                        if j>i: # Avoid double checking
                            client_wise_dist += np.linalg.norm( clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key] )
#                         new_models[i][key] += avg_weight * adj[i,j] * clients[j].model.state_dict()[key]
                    
                    dists[i,j,k,e] = client_wise_dist
                    
                    if client_wise_dist > avg_error_thres and reached_consensus and j>i:
                        reached_consensus = False
            if reached_consensus:
                break
            
            # Process the recorded values
            if avg_alg == 'epsilon':
                Ni = np.min( np.sum(adj, axis=0) )
                avg_weight = min(avg_weight, 0.99/Ni)
                for i in range(len(clients)):
                    for j in range(len(clients)):
                        for key in new_models[j].keys():
                            new_models[i][key] += avg_weight * adj[i,j] * (clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key])
            elif avg_alg == 'metropolis':
                Nis = np.sum(adj, axis=0)
                for i in range(len(clients)):
                    for j in range(len(clients)):
                        for key in new_models[j].keys():
                            new_models[i][key] += avg_weight * adj[i,j] * (clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key]) / max(Nis[i], Nis[j])
            else:
                for i in range(len(clients)):
                    Ni = np.sum(adj[i,:])
                    # Calcualte the averages, or use other methods
                    for j in range(len(clients)):
                        for key in new_models[j].keys():
                            new_models[i][key] += avg_weight * adj[i,j] * clients[j].model.state_dict()[key]
                    for key in new_models[i].keys():
#                         new_models[i][key] -= clients[i].model.state_dict()[key]
                        new_models[i][key] /= (Ni+1) # Or use torch.div()

            # Load averaged results
            for i in range(len(clients)):
                clients[i].model.load_state_dict(new_models[i])
        print("Finished averaging at the {}-th iteration".format(k))
        # Fill the rest of the dists array with the last distance info after breaking out
        while k < dists.shape[2]:
            for i in range(len(clients)):
                for j in range(i+1,len(clients)):
                    dists[i,j,k,e] = dists[i,j,k-1,e]
            k += 1
        
        # Run the client with lowest loss against test dataset to examine performance
        if test_loader is not None:
            if test_all_client:
                test_loss.append([])
                saccus = []
                # Let all clients go over the test, and take the max accuracy as this round's accuracy
                for c in clients:
                    sloss, saccu = test_DFL([c], test_loader, debug=False)
                    test_loss[-1].append(sloss)
                    saccus.append(saccu)
                test_accuracy.append(np.max(saccu))
            else:
                # Only record the best client's performance
                sloss, saccu = test_DFL(clients, test_loader, debug=False)
                test_loss.append([sloss])
                test_accuracy.append(saccu)
        print("round {0} complete with {1} averaging steps and test accuracy {2}".format(e+1,k,test_accuracy[-1]))
        
        # Preemptively save the parameters and the clients
        if bother_saving > 0 and (e % bother_saving == 0):
            with open(save_model_to, "wb") as fp:
                pickle.dump([test_loss, test_accuracy, e], fp)
            for i,c in enumerate(clients):
                torch.save(c.model.state_dict(), save_model_to+'_client{}'.format(i))
        
    return test_loss, test_accuracy
    
def test_DFL(clients, test_loader, debug=True):
    # Pick the client with the lowest loss to run against test dataset
    losses = []
    for c in clients:
        c.model.eval()
        losses.append(np.mean(c.loss_history[-1]))
    ind = np.argmin(losses)
    server = clients[ind]

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = server.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if debug:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

# Define some parameters for DFL
K = 100 # Number of clients
C = 1
B = 64#10 # Batch size in clients
E = 1 # Epochs per client update (>= 1)
n = len(dataset1) # Number of training samples in total
num_rounds = 500#200
EE = 1/20 # (1/num_rounds) # Portion of data per epoch (<= 1)
max_n = n/K * EE

nk = [int(n/K)]*K
nk[-1] = n - sum(nk[:-1])
print(nk)

# Create adjacency matrix
AK = np.zeros((K,K))
# For a complete graph:
AK = np.ones((K,K))
for i in range(K):
    AK[i,i] = 0
AC = np.zeros((K,K))
AC[0,-1] = 1
AC[-1,0] = 1
for i in range(K-1):
    AC[i,i+1] = 1
    AC[i+1,i] = 1

# Create agents and assign data to each agent
partitioned_dataset = torch.utils.data.random_split(dataset1, nk)

# Create agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(partitioned_dataset[i], B, E))
#     clients_DFL.append(ClientNet(dataset1, B, E))

# Optional: Set all clients to the same initial weights
for i in range(1,K):
    clients_DFL[i].model.load_state_dict( clients_DFL[0].model.state_dict() )

script_filiename = 'MNIST_DFL_noconsensus_split_dataset_K{}_C{}_E{}_'.format(K,C,E*EE)
script_filename = 'tmp_data/'+script_filiename

# Prepare some files for backup purposes.
# Note: If we want to continue where we left off, then it's reasonable to assume that we know what's the settings
# from the previous run. In other words, we assume that the variables above of this section are still the same. 
# Thus, if you try to load things with different settings, expect unexpected behaviors.
file_identifier = ''
if len(sys.argv) > 1:
    file_identifier = sys.argv[1]
try:
    with open(script_filename+file_identifier, 'rb') as fp:
        params = pickle.load(fp)
    for i,cl in enumerate(clients_DFL):
        cl.model.load_state_dict(torch.load(script_filename+file_identifier+'_client{}'.format(i)))
except:
    params = None
    print("Did not find file that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")

num_avg_iter = 1
dists = np.zeros((K,K,num_avg_iter,num_rounds))
DFL_loss_history, DFL_accuracy_history = train_DFL(AK, clients_DFL, dists, num_rounds=num_rounds, dry_run=False, 
                                                   CK=K, max_n=max_n, 
                             test_loader=test_loader, test_all_client=True,
              num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01,
              past_params=params, save_model_to=script_filename+file_identifier, bother_saving=10)#, 
#                                                   avg_alg='epsilon', avg_weight=0.05)
test_DFL(clients_DFL, test_loader)

for i in range(K):
    np.save(script_filiename+'client_{0}_loss_history'.format(i), clients_DFL[i].loss_history)
np.save(script_filiename+'Accuracy', DFL_accuracy_history)
np.save(script_filiename+'average_loss_history',DFL_loss_history)
np.save(script_filiename+'dists_history',dists)

# # Do it again!

# K = 30 # Number of clients
# C = 1
# B = 64#10 # Batch size in clients
# E = 1 # Epochs per client update (>= 1)
# n = len(dataset1) # Number of training samples in total
# num_rounds = 150
# EE = 1/20 # (1/num_rounds) # Portion of data per epoch (<= 1)
# max_n = n/K * EE

# nk = [int(n/K)]*K
# nk[-1] = n - sum(nk[:-1])
# print(nk)

# # Create adjacency matrix
# AK = np.zeros((K,K))
# # For a complete graph:
# AK = np.ones((K,K))
# for i in range(K):
#     AK[i,i] = 0

# # Create agents and assign data to each agent
# partitioned_dataset = torch.utils.data.random_split(dataset1, nk)

# # Create agents
# clients_DFL = []
# for i in range(K):
#     clients_DFL.append(ClientNet(partitioned_dataset[i], B, E))
# #     clients_DFL.append(ClientNet(dataset1, B, E))

# # Optional: Set all clients to the same initial weights
# for i in range(1,K):
#     clients_DFL[i].model.load_state_dict( clients_DFL[0].model.state_dict() )

# num_avg_iter = 1
# dists = np.zeros((K,K,num_avg_iter,num_rounds))
# DFL_loss_history, DFL_accuracy_history = train_DFL(AK, clients_DFL, dists, num_rounds=num_rounds, dry_run=False, 
#                                                    CK=K, max_n=max_n, 
#                              test_loader=test_loader, test_all_client=True,
#               num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01)
# test_DFL(clients_DFL, test_loader)

# script_filiename = 'MNIST_DFL_noconsensus_split_dataset_K{}_C{}_E{}_'.format(K,C,E*EE)
# for i in range(K):
#     np.save(script_filiename+'client_{0}_loss_history'.format(i), clients_DFL[i].loss_history)
# np.save(script_filiename+'Accuracy', DFL_accuracy_history)
# np.save(script_filiename+'average_loss_history',DFL_loss_history)
# np.save(script_filiename+'dists_history',dists)


# Code for proof-of-concept testing
# backupclient = ClientNet(partitioned_dataset[i], B, E)
# backupclient.model.load_state_dict( clients_DFL[0].model.state_dict() )
# DFL_loss_history, DFL_accuracy_history = train_DFL(AC, clients_DFL, dists, num_rounds=num_rounds, dry_run=True, 
#                                                    CK=K, max_n=max_n, 
#                              test_loader=test_loader, test_all_client=False, avg_alg='epsilon', avg_weight=0.05,
#               num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01)
# np.save('ncsns_epsilon_dists_history',dists)
# 
# dists = np.zeros((K,K,num_avg_iter,num_rounds))
# for i in range(K):
#     clients_DFL[i].model.load_state_dict( backupclient.model.state_dict() )
# DFL_loss_history, DFL_accuracy_history = train_DFL(AC, clients_DFL, dists, num_rounds=num_rounds, dry_run=True, 
#                                                    CK=K, max_n=max_n, 
#                              test_loader=test_loader, test_all_client=False, avg_alg='metropolis',
#               num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01)
# np.save('ncsns_metropolis_dists_history',dists)
# 
# dists = np.zeros((K,K,num_avg_iter,num_rounds))
# for i in range(K):
#     clients_DFL[i].model.load_state_dict( backupclient.model.state_dict() )
# DFL_loss_history, DFL_accuracy_history = train_DFL(AC, clients_DFL, dists, num_rounds=num_rounds, dry_run=True, 
#                                                    CK=K, max_n=max_n, 
#                              test_loader=test_loader, test_all_client=False,
#               num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01)
# np.save('ncsns_cycle_dists_history',dists)
