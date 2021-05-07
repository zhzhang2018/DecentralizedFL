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
        return x
#         output = F.log_softmax(x, dim=1)
#         return output
    
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
    def __init__(self, dataset, B=16, E=1, optimizerlr=1, schedgamma=0.7, optim_method=optim.Adadelta, loss_f=nn.CrossEntropyLoss):
        self.model = Net().to(device)
        self.optimizer = optim_method(self.model.parameters(), lr=optimizerlr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=schedgamma)
        self.loss_f = loss_f()
        self.loss_history = []
        self.B = B
        self.E = E
        train_kwargs = {'batch_size': B, 'shuffle': True}
        self.train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)
        if loss_f == nn.CrossEntropyLoss:
            self.output_processing = lambda x : x
        elif loss_f == nn.BCEWithLogitsLoss:
            self.output_processing = lambda x : (F.one_hot(x).to(torch.float))
        
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
                output = self.output_processing(output)
                loss = self.loss_f(output, target)
                loss.backward()
                self.optimizer.step()

                self.loss_history[-1].append(loss.item())

                if dry_run or batch_idx*self.B >= max_n:
                    break
#         print(len(self.loss_history[-1]))
        
# Try to see if you can train distributedly - i.e. average between several separate models
# Implement train and test logic. This one doens't have a server, and I assume we can let them all update together.
# dists: expected to be a K x K x num_avg_iter x num_rounds array to store distances
def train_DFL(adj, clients, dists, num_rounds=1, dry_run=False, debug=False, CK=10, max_n=np.inf, test_loaders=None,
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
        if test_loaders is not None:
            if test_all_client and len(clients) == len(test_loaders):
                test_loss.append([])
                saccus = []
                # Let all clients go over the test, and take the max accuracy as this round's accuracy
                for i,c in enumerate(clients):
                    sloss, saccu = test_DFL([c], test_loaders[i], debug=False)
                    test_loss[-1].append(sloss)
                    saccus.append(saccu)
                test_accuracy.append(saccus)
#                 test_accuracy.append(np.max(saccu))
            else:
                # Only record the best client's performance
                sloss, saccu = test_DFL(clients, test_loaders, debug=False)
                test_loss.append([sloss])
                test_accuracy.append(saccu)

            # Detect past accuracy records and decide if you still want to continue training.
            # Tentative stopping criterion: If 1) accuracy is too low, or 2) accuracy is decreasing.
            # The stopping criterion is checked with multiple historical values to reduce false positives.
            if len(test_accuracy) > 50:
                break_bc_low = False
                break_bc_drop = False
                if (( np.mean(test_accuracy[-1]) < np.mean(test_accuracy[-50])-5 ) and 
                   ( np.mean(test_accuracy[-20]) < np.mean(test_accuracy[-50])-3 ) and 
                   ( np.mean(test_accuracy[-5]) < np.mean(test_accuracy[-50])-5 ) and 
                   ( np.mean(test_accuracy[-1]) < 90 )):
                    break_bc_drop = True
                if (( np.mean(test_accuracy[-1]) < 50 ) and 
                   ( np.mean(test_accuracy[-20]) < 50 ) and 
                   ( np.mean(test_accuracy[-50]) < 70 )):
                    break_bc_low = True
                if break_bc_drop:# or break_bc_low:
                    break
            
        print("round {0} complete with {1} averaging steps and test accuracy {2}".format(e+1,k,test_accuracy[-1]))
        
        # Preemptively save the parameters and the clients
        if bother_saving > 0 and ((e+1) % bother_saving == 0):
            with open(save_model_to, "wb") as fp:
                pickle.dump([test_loss, test_accuracy, e], fp)
            for i,c in enumerate(clients):
                torch.save(c.model.state_dict(), save_model_to+'_client{}'.format(i))
                with open(save_model_to+'_client{}_loss_history'.format(i), "wb") as fp:
                    pickle.dump(c.loss_history, fp)
            np.save(script_filename+file_identifier+'_dists_history_temp.npy', dists)
        
    return test_loss, test_accuracy
    
def test_DFL(clients, test_loaders, debug=True):
    # Pick the client with the lowest loss to run against test dataset
    losses = []
    for c in clients:
        c.model.eval()
#         print(c.loss_history)
        losses.append(np.mean(c.loss_history[-1]))
    ind = np.argmin(losses)
    server = clients[ind]

    # MODIFIED test_DFL method!
    # If test_loaders is a single loader, then use it for all given clients. 
    # If test_loaders is a list of loaders, then use each for each client.
    test_loss = 0
    correct = 0
    testDatasetSize = 0
    with torch.no_grad():
        if type(test_loaders) == torch.utils.data.dataloader.DataLoader:
#            print("Dealing with single loader")
            for data, target in test_loaders:
                data, target = data.to(device), target.to(device)
                output = server.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            testDatasetSize = len(test_loaders.dataset)
            test_loss /= testDatasetSize
        else:
#            print("Assuming dealing with a list now")
            for i in range(len(test_loaders)):
                for data, target in test_loaders[i]:
                    data, target = data.to(device), target.to(device)
                    output = clients[i].model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                testDatasetSize += len(test_loaders[i].dataset)
            test_loss /= testDatasetSize
    if debug:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, testDatasetSize,
            100. * correct / testDatasetSize))
    return test_loss, 100. * correct / testDatasetSize

# Define some parameters for DFL
K = 10 # Number of clients
C = 1
B = 64#10 # Batch size in clients
E = 1 # Epochs per client update (>= 1)
n = len(dataset1) # Number of training samples in total
num_rounds = 55
EE = 1/20 # (1/num_rounds) # Portion of data per epoch (<= 1)
max_n = n/K * EE

import utils
S = 1.0
Mean = 1
V = 0.7
nk = utils.generate_partition_sizes(K, n, mean=Mean, variance=V, options=None, minsize=B)

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

loss_fs = [nn.CrossEntropyLoss] * K
list_size_counts = [K]

file_type = 'A'
if len(sys.argv) > 2:
    file_type = sys.argv[2]
### Format of file_type: (X[0-9]^2)[A-F](other file type codes)[lower case indicator]
if file_type[0] == 'X':
    # X stands for miXing Xross-entropy-loss with Binary-Xross-entropy-loss-with-logits
    bxelwl_count = int(file_type[1:3])
    file_type = file_type[3:]
    loss_fs = [nn.BCEWithLogitsLoss] * bxelwl_count + [nn.CrossEntropyLoss] * (K-bxelwl_count)
if file_type[0] == 'F':
    if file_type[-1] == 'm':
        list_size_counts = [K-1, 1]
        file_type = file_type[:-1]
    elif file_type[-1] == 'n':
        list_size_counts = [K-3, 3]
        file_type = file_type[:-1]
    elif file_type[-1] == 's':
        list_size_counts = [K-5, 5]
        file_type = file_type[:-1]

if file_type[0] == 'A':
    lr_list = [0.001] * K
    if file_type == 'A2':
        lr_list = [0.01] * K # Bad values!
    elif file_type == 'A15':
        lr_list = [0.005] * K  # Bad with waves; 98% when starting uniform
    elif file_type == 'Ao1':
        lr_list = [0.001] * (K-1) + [0.01] # OK
    elif file_type == 'Ao3':
        lr_list = [0.001] * (K-3) + [0.01] * 3 # 95.4%
    elif file_type == 'Ao5':
        lr_list = [0.001] * (K-5) + [0.01] * 5 # Bad; 98% when starting uniform
    elif file_type == 'Ac1':
        lr_list = 0.001 * np.arange(1,K+1) # Bad bad bad bad bad
    elif file_type == 'Ac15':
        lr_list = 0.001 + 0.0005 * np.arange(1,K+1) # Bad
    optim_methods = [optim.Adam] * K
elif file_type[0] == 'B':
    lr_list = [5] * K # worked but not too stable
    if file_type == 'B1':
        lr_list = [1] * K # Baseline vaseline
    optim_methods = [optim.Adadelta] * K
elif file_type[0] == 'C':
    lr_list = [0.01] * K # Bad value; 97.6% when starting uniform
    if file_type == 'C3':
        lr_list = [0.001] * K # 92.74%
    elif file_type == 'Cc1':
        lr_list = 0.001 * np.arange(1,K+1) # 96%
    elif file_type == 'Cc2':
        lr_list = 0.002 * np.arange(1,K+1) # 96%
    # elif file_type == 'Co3':
    #     lr_list = [0.001] * (K-3) + [0.01] * 3 # Bad; 97.5% when starting uniform but has noise
    optim_methods = [optim.Adagrad] * K
elif file_type[0] == 'D':
    lr_list = [0.01] * K # Bad value
    if file_type == 'D3':
        lr_list = [0.001] * K # Still did not work; 98% when starting uniform
    elif file_type == 'D4':
        lr_list = [0.0001] * K # 94%
    elif file_type == 'D45':
        lr_list = [0.0005] * K # 96.9%
    elif file_type == 'Dc1':
        lr_list = 0.001 * np.arange(1,K+1) # Did not work even with uniform start
    optim_methods = [optim.RMSprop] * K
elif file_type[0] == 'E':
    if file_type == 'E0':
        lr_list = [1] * K # Bad value
    elif file_type == 'E1':
        lr_list = [0.1] * K # Naughtily sensitive
    elif file_type == 'E2':
        lr_list = [0.01] * K # Naughtily sensitive
    elif file_type == 'E3':
        lr_list = [0.001] * K # Very slow
    elif file_type == 'Ec1':
        lr_list = 0.1 * np.arange(1,K+1) # Bad
    elif file_type == 'Ec2':
        lr_list = 0.01 * np.arange(1,K+1) # Noisy 95.5%
    elif file_type == 'Ec12':
        lr_list = 0.01 * np.array([1,3,5,7,9,20,40,60,80,100]) # 96.83% Very noisy
    optim_methods = [optim.SGD] * K
# F stands for Fusion
elif file_type[0] == 'F':
    if file_type == 'F_AC': 
        lr_list = [0.001] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        #m: 96.65%; Uniform initial weights: 97.88%
        # n: 97.02%; Uniform initial weights: %
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_AD': 
        lr_list = [0.001] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        #m: 97.32%; Uniform initial weights: 97.83%
        # n: 97.38%; Uniform initial weights: %
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
    elif file_type == 'F_AE': 
        lr_list = [0.001] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        #m: 96.91%; Uniform initial weights: 97.89%
        # n: 96.88%; Uniform initial weights: %
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_AB': 
        lr_list = [0.001] * list_size_counts[0] + [0.01] * list_size_counts[1] 
        #m: 96.9%; Uniform initial weights: 97.72%
        # n: 96.37% "diverging loss" for the last 3; Uniform initial weights: %
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.Adadelta] * list_size_counts[1] 
    elif file_type == 'F_CA': 
        lr_list = [0.005] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        #m: 95.86%; Uniform initial weights: 97.54%
        # n: 96.11%; Uniform initial weights: %
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_CD': 
        lr_list = [0.005] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        #m: 94.45%; Uniform initial weights: 97.64%
        # n: 96.41%; Uniform initial weights: %
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
    elif file_type == 'F_CE': 
        lr_list = [0.005] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        #m: 96.43%; Uniform initial weights: 97.64%
        # n: %; Uniform initial weights: %
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_DA': 
        lr_list = [0.0005] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        #m: 97.1%; Uniform initial weights: 97.77%
        # n: 97.29%; Uniform initial weights: %
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_DC': 
        lr_list = [0.0005] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        #m: 96.88%; Uniform initial weights: 97.94%
        # n: 95.96%; Uniform initial weights: %
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_DE': 
        lr_list = [0.0005] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        #m: 96.59%; Uniform initial weights: 97.47%
        # n: 97.28%; Uniform initial weights: %
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_EA': 
        lr_list = [0.05] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        #m: 95.58%; Uniform initial weights: 97.26%
        # n: 96.14%; Uniform initial weights: %
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_EB': 
        lr_list = [0.05] * list_size_counts[0] + [0.01] * list_size_counts[1] 
        #m: 95.35%; Uniform initial weights: 96.93%
        # n: 94.84% unsteady start; Uniform initial weights: %
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adadelta] * list_size_counts[1] 
    elif file_type == 'F_EC': 
        lr_list = [0.05] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        #m: 96% with peak; Uniform initial weights: 97.3%
        # n: %; Uniform initial weights: %
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_ED': 
        lr_list = [0.05] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        #m: 96.09% with peak; Uniform initial weights: 97.25%
        # n: 96.86%; Uniform initial weights: %
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
# G stands for Group Fusion
#elif 
else:
    file_type = 'BassLine'
    lr_list = [1] * K
    optim_methods = [optim.Adadelta] * K
    
# Attach "SameInit" to the end of file_type if specified
SameInit = False
if len(sys.argv) > 2:
    file_type = sys.argv[2]
if len(sys.argv) > 3 and sys.argv[3] == "-S":
    SameInit = True
if SameInit:
    file_type += 'SameInit'

if len(sys.argv) > 2 and sys.argv[-2] == "-s":
    S = float(sys.argv[-1])

use_unbalanced_dataset = True#False
unb_name = ''
if use_unbalanced_dataset:
    unb_name = 'Unbalanced'

script_filiename = 'MNIST_DFL_diverseNorm{}_{}dataset_K{}_C{}_E{}_S{}_M{}_V{}_'.format(file_type,unb_name,K,C,E*EE,S,Mean,V)
#script_filename = 'tmp_data/'+script_filiename
script_filename = '../../../../../../scratch1/7/zzhang433/tmp_data/'+script_filiename
# Prepare some files for backup purposes.
# Note: If we want to continue where we left off, then it's reasonable to assume that we know what's the settings
# from the previous run. In other words, we assume that the variables above of this section are still the same. 
# Thus, if you try to load things with different settings, expect unexpected behaviors.
file_identifier = ''
if len(sys.argv) > 1:
    file_identifier = sys.argv[1]
    
num_avg_iter = 100
loaded_prev_session = False
try:
    with open(script_filename+file_identifier, 'rb') as fp:
        params = pickle.load(fp)
    with open(script_filename+file_identifier+'_split', 'rb') as fp:
        nk = pickle.load(fp)
    loaded_prev_session = True
except:
    params = None
    print("Did not find file that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")
    with open(script_filename+file_identifier+'_split', 'wb') as fp:
        pickle.dump(nk, fp)

# Create agents and assign data to each agent.
# IMPORTANT: If you pick up from where you left off, then you MUST keep the seed below the same.
# Otherwise, the training data for each client is not consistent between sessions.
print(nk)
nshare = [int(nsample / (K-1) * S) for nsample in nk]

torch.manual_seed(2021)
# Optional: You can also use datasets with only one label across
if use_unbalanced_dataset:
    # Categorize training data by label
    datasets_by_label = []
    for i in range(10):
        inds = [j for j in range(len(dataset1)) if dataset1.targets[j] == i]
        datasets_by_label.append(copy.deepcopy(dataset1))
        datasets_by_label[-1].data = datasets_by_label[-1].data[np.r_[inds]]
        datasets_by_label[-1].targets = datasets_by_label[-1].targets[np.r_[inds]]
    
    # Assign data to different clients to cater for the partition.
    # The easiest way is when nk isn't randomly selected, as implemented below
    nk = [len(dts) for dts in datasets_by_label]
    nshare = [int(nsample / (K-1) * S) for nsample in nk]
    partitioned_full_dataset = datasets_by_label
    print("Had to modify the dataset partition to create unbalanced dataset. But still, it is done.")
    print(nk)
    print("Looking into each partition's statistics:")
    for i in range(10):
        print("Class {}: ".format(i), datasets_by_label[i].data.float().mean(), 
              datasets_by_label[i].data.float().std())
    print("Overall: 0.1307, 0.3081?")
else:
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

# Set the normalization parameters for each client
use_same_base_norm = False#True#False#True
norm_mean = 0.1307
norm_stdev = 0.3081
norm_records = []
test_dataset_loaders = []
if use_same_base_norm:
    # Option 1: Use custom-determined means and stdev values
    for i in range(K):
        # Method 1: Deduct a certain value from the first half, and increase for the second half
        norm_records.append( (norm_mean + ((i // (int(K/2)-1)) - 0.5)*2 * 0, 
                              norm_stdev+ ((i // (int(K/2)-1)) - 0.5)*2 * 0) )
        # Method 2: Use random
#         norm_records.append( (norm_mean + (np.random.random()*2 - 1) * 0.1, 
#                               norm_stdev+ (np.random.random()*2 - 1) * 0.01) )
else:
    # Option 2: Use the mean and stdev from each dataset
    for i in range(K):
#        print(partitioned_dataset[i].datasets)
        norm_records.append( (partitioned_full_dataset[i].data.float().mean()/255.0, 
                              partitioned_full_dataset[i].data.float().std()/255.0) )
print(norm_records)
# And eventually you'll need to set their dataset's transform
for i in range(K):
    test_set_i = copy.deepcopy(dataset2)
    test_set_i.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_records[i][0],), (norm_records[i][1],))
    ])
    test_dataset_loaders.append(torch.utils.data.DataLoader(test_set_i, **test_kwargs))
    partitioned_dataset[i].transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_records[i][0],), (norm_records[i][1],))
    ])
print("Resulting normalization factors:", norm_records)

torch.manual_seed(seed) # Return to previous seed

Es = [E]*K # baseline (b
# Es = [E]*(K-1)+[E*2] # Mild baseline (a
# Es = [4*E]*(K//2)+[E]*(K-K//2) # Half normal and half fast (c
# Es = [5*E]*(K-3)+[E,E,E] # Most are faster (d
#Es = np.arange(1,K+1, dtype=int)*E*0.5 # Continuum 1 to 10 (e

# Create agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(partitioned_dataset[i], B, E=Es[i], optimizerlr=lr_list[i], 
                                 optim_method=optim_methods[i], loss_f=loss_fs[i]))

try:
    for i,cl in enumerate(clients_DFL):
        cl.model.load_state_dict(torch.load(script_filename+file_identifier+'_client{}'.format(i)))
        with open(script_filename+file_identifier+'_client{}_loss_history'.format(i), "rb") as fp:
            cl.loss_history = pickle.load(fp)
except:
    print("Did not find model weights that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")
    # Optional: Set all clients to the same initial weights 
    if SameInit:
        for i in range(1,K):
            clients_DFL[i].model.load_state_dict( clients_DFL[0].model.state_dict() )
try:
    dists = np.load(script_filename+file_identifier+'_dists_history_temp.npy')
except:
    print("Did not find model distances that are called '"+script_filename+file_identifier+"_dists_history_temp.npy'. Starting from zeros.")
    dists = np.zeros((K,K,num_avg_iter,num_rounds))
if (dists.shape[3] < num_rounds):
    dists = np.concatenate((dists, np.zeros((K,K,dists.shape[2],num_rounds-dists.shape[3]))), axis=3)
    print("Extended dists into shape: ", dists.shape)
'''
try:
    for i,cl in enumerate(clients_DFL):
        cl.model.load_state_dict(torch.load(script_filename+file_identifier+'_client{}'.format(i)))
        with open(script_filename+file_identifier+'_client{}_loss_history'.format(i), "rb") as fp:
            cl.loss_history = pickle.load(fp)
    dists = np.load(script_filename+file_identifier+'_dists_history_temp.npy')
    if (dists.shape[3] < num_rounds):
        dists = np.concatenate((dists, np.zeros(K,K,dists.shape[2],num_rounds-dists.shape[3])), axis=3)
        print("Extended dists into shape: ", dists.shape)
# else:
except:
    print("Did not find model weights that starts with '"+script_filename+file_identifier+"'. Starting from scratch.")
    # Optional: Set all clients to the same initial weights
    if SameInit:
        for i in range(1,K):
            clients_DFL[i].model.load_state_dict( clients_DFL[0].model.state_dict() )
    dists = np.zeros((K,K,num_avg_iter,num_rounds))
'''
DFL_loss_history, DFL_accuracy_history = train_DFL(AK, clients_DFL, dists, num_rounds=num_rounds, dry_run=False, 
                                                   CK=K, max_n=max_n, 
                             test_loaders=test_dataset_loaders, test_all_client=True,
              num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=0.01,
              past_params=params, save_model_to=script_filename+file_identifier, bother_saving=10, 
                                                   avg_alg='epsilon', avg_weight=0.05)
test_DFL(clients_DFL, test_dataset_loaders)

for i in range(K):
    np.save(script_filiename+'client_{0}_loss_history'.format(i), clients_DFL[i].loss_history)
np.save(script_filiename+'Accuracy', DFL_accuracy_history)
np.save(script_filiename+'average_loss_history',DFL_loss_history)
np.save(script_filiename+'dists_history',dists[:,:,[0,-1],:])
