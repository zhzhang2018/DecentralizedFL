# This file continues separated, individual training, and checks the test results in more detail
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
        # No more aggregation for each client. 
        print("Finished updating for {}-th round".format(e))
        # But still, we can check their distances after each epoch.
        for i in range(len(clients)):
            for j in range(len(clients)):
                client_wise_dist = 0
                for key in clients[j].model.state_dict().keys():
                    if j>i: # Avoid double checking
                        client_wise_dist += np.linalg.norm( clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key] )
                dists[i,j,0,e] = client_wise_dist
        
        # Run against test dataset to examine performance
        if test_loader is not None:
            if test_all_client:
                test_loss.append([])
                saccus = []
                # Let all clients go over the test, and take the max accuracy as this round's accuracy
                for c in clients:
                    sloss, saccu = test_DFL([c], test_loader, debug=False)
                    test_loss[-1].append(sloss)
                    saccus.append(saccu)
                test_accuracy.append(saccus)
#                 test_accuracy.append(np.max(saccu))
            else:
                # Only record the best client's performance
                sloss, saccu = test_DFL(clients, test_loader, debug=False)
                test_loss.append([sloss])
                test_accuracy.append(saccu)
        print("round {0} complete with test accuracy {1}".format(e+1,test_accuracy[-1]))
        
        # Preemptively save the parameters and the clients
        if bother_saving > 0 and (e % bother_saving == 0):
            with open(save_model_to, "wb") as fp:
                pickle.dump([test_loss, test_accuracy, e], fp)
            for i,c in enumerate(clients):
                torch.save(c.model.state_dict(), save_model_to+'_client{}'.format(i))
                with open(save_model_to+'_client{}_loss_history'.format(i), "wb") as fp:
                    pickle.dump(c.loss_history, fp)
            np.save(script_filename+file_identifier+'_dists_history_temp.npy', dists)
        
    return test_loss, test_accuracy
    
def test_DFL(clients, test_loader, debug=True, return_preds=False):
    # Pick the client with the lowest loss to run against test dataset
    losses = []
    for c in clients:
        c.model.eval()
        losses.append(np.mean(c.loss_history[-1]))
    ind = np.argmin(losses)
    server = clients[ind]

    test_loss = 0
    correct = 0
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = server.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.append([pred.numpy(), target.view_as(pred).numpy()])

    test_loss /= len(test_loader.dataset)
    if debug:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    if return_preds:
        return test_loss, 100. * correct / len(test_loader.dataset), preds
    
    return test_loss, 100. * correct / len(test_loader.dataset)

# Define some parameters for DFL
K = 10 # Number of clients
C = 1
B = 64#10 # Batch size in clients
E = 1 # Epochs per client update (>= 1)
n = len(dataset1) # Number of training samples in total
num_rounds = 50
EE = 1/20 # (1/num_rounds) # Portion of data per epoch (<= 1)
max_n = n/K * EE

import utils
S = 1.0
Mean = 1
V = 0.7
nk = utils.generate_partition_sizes(K, n, mean=Mean, variance=V, options=None, minsize=B)

# Create adjacency matrix
AK = np.zeros((K,K))

codetype = 'full'
script_filiename = 'MNIST_DFL_{}_dataset_K{}_C{}_E{}_S{}_M{}_V{}_'.format(codetype,K,C,E*EE,S,Mean,V)
file_identifier = ''
if len(sys.argv) > 1:
    file_identifier = sys.argv[1]
if len(sys.argv) > 2:
    script_filiename = sys.argv[2]
script_filename = 'tmp_data/'+script_filiename

try:
    with open(script_filename+file_identifier, 'rb') as fp:
        params = pickle.load(fp)
    with open(script_filename+file_identifier+'_split', 'rb') as fp:
        nk = pickle.load(fp)
except:
    print(("Did not find some of the files that starts with '"+script_filename+file_identifier+"'. Might be troublesome."))
#     sys.exit("Did not find file that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")

nshare = [int(nsample / (K-1) * S) for nsample in nk]

torch.manual_seed(2021)
partitioned_full_dataset = torch.utils.data.random_split(dataset1, nk)
partitioned_dataset = []
for i in range(K):
    partitioned_dataset.append(partitioned_full_dataset[i])
    for j in range(K):
        if i != j:
            j_shared_dataset, trivial_ = torch.utils.data.random_split(
                partitioned_full_dataset[j], [nshare[j], nk[j]-nshare[j]]
            )
            partitioned_dataset[-1] = partitioned_dataset[-1] + j_shared_dataset
print([len(pd) for pd in partitioned_dataset])
torch.manual_seed(seed) # Return to previous seed

lr_list = [lr]*K # Case 0 baseline

# Create agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(partitioned_dataset[i], B, E=E, optimizerlr=lr_list[i]))

try:
    for i,cl in enumerate(clients_DFL):
        cl.model.load_state_dict(torch.load(script_filename+file_identifier+'_client{}'.format(i)))
        with open(script_filename+file_identifier+'_client{}_loss_history'.format(i), "rb") as fp:
            cl.loss_history = pickle.load(fp)
    dists = np.load(script_filename+file_identifier+'_dists_history_temp.npy')
except:
    sys.exit("Did not find model weights that starts with '"+script_filename+file_identifier)

dists = np.concatenate(( dists[:,:,[-1],:], np.zeros((K,K,1,num_rounds)) ), axis=3)
params[2] = dists.shape[3]

_, _, prior_preds = test_DFL(clients_DFL, test_loader, return_preds=True)
np.save(script_filiename+'_test_preds_rightafter.npy', prior_preds)

DFL_loss_history, DFL_accuracy_history = train_DFL(AK, clients_DFL, dists, 
                                    num_rounds=num_rounds+params[2], dry_run=False, 
                                CK=K, max_n=max_n, 
                             test_loader=test_loader, test_all_client=True,
              past_params=params, save_model_to=script_filename+file_identifier, bother_saving=10)
_, _, after_preds = test_DFL(clients_DFL, test_loader, return_preds=True)
np.save(script_filiename+'_test_preds_final.npy', after_preds)

script_filiename += '_INDIV_'
for i in range(K):
    np.save(script_filiename+'client_{0}_loss_history'.format(i), clients_DFL[i].loss_history)
np.save(script_filiename+'Accuracy', DFL_accuracy_history)
np.save(script_filiename+'average_loss_history',DFL_loss_history)
np.save(script_filiename+'dists_history',dists)
