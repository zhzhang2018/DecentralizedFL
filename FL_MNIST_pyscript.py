# Official training implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

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

def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run, debug=False):
    model.train()
    loss_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if debug:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break
        loss_history.append(loss.item())
    return loss_history

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    
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

# Main execution for official training
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

loss_history = []
for epoch in range(1, epochs + 1):
    loss_history.append(
        train(model, device, train_loader, optimizer, epoch, log_interval, dry_run, debug=False)
    )
    print('Finished epoch {0} with loss {1}'.format(epoch, loss_history[-1][-1]))
    test(model, device, test_loader)
    scheduler.step()

if save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
    
    
    
    
# Standard FL
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

                if dry_run or batch_idx >= max_n:
                    break
        
# Implement train and test logic
def train_FL(server, clients, num_rounds=1, dry_run=False, debug=False, CK=10, max_n=np.inf, test_loader=None):
    server_loss = []
    server_accuracy = []
    for e in range(num_rounds):
        client_inds = np.random.permutation(len(clients))
        chosen_clients = []
        for m in client_inds[:CK]:
            clients[m].update(dry_run, max_n=max_n, weights=server.model.state_dict())
            chosen_clients.append(clients[m])
        server.update(chosen_clients)
        print("round {0} complete".format(e+1))
        # Run the server against test dataset to examine performance
        if test_loader is not None:
            sloss, saccu = test_FL(server, test_loader, debug=False)
            server_loss.append([sloss])
            server_accuracy.append(saccu)
    return server_loss, server_accuracy
    
def test_FL(server, test_loader, debug=True):
    server.model.eval()
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

# Define some parameters for FL
K = 20 # Number of clients
C = 0.5 # Fraction of clients updated each round
CK = int(C*K)
B = 10 # Batch size in clients
E = 1 # Epochs per client update
n = len(dataset1) # Number of training samples in total
num_rounds = 100
max_n = n/num_rounds/CK
nk = [int(n/K)]*K
nk[-1] = n - np.sum(nk[:-1])

# Create agents and assign data to each agent
partitioned_dataset = torch.utils.data.random_split(dataset1, nk)

# Create agents
clients = []
for i in range(K):
#     clients.append(ClientNet(dataset1, B, E))
    clients.append(ClientNet(partitioned_dataset[i], B, E))
server = ServerNet()

server_loss_history, server_accuracy_history = train_FL(server, clients, num_rounds=num_rounds, dry_run=False, CK=CK, max_n=max_n, test_loader=test_loader)
test_FL(server, test_loader)




# Decentralized FL
# Try to see if you can train distributedly - i.e. average between several separate models
# Implement train and test logic. This one doens't have a server, and I assume we can let them all update together.
def train_DFL(adj, clients, num_rounds=1, dry_run=False, debug=False, CK=10, max_n=np.inf, test_loader=None,
              avg_weight=1):
    test_loss = []
    test_accuracy = []
    for e in range(num_rounds):
        # Update each client
        for m in clients:
            m.update(dry_run, max_n=max_n)
        # Aggregation for each client
        new_models = []
        for i in range(len(clients)):
            new_models.append( copy.deepcopy(clients[i].model.state_dict()) )
            Ni = np.sum(adj[i,:])
            for key in new_models[-1].keys():
                for j in range(len(clients)):
                    new_models[-1][key] += avg_weight * adj[i,j] / Ni * clients[j].model.state_dict()[key]
#                 new_models[-1][key] = torch.div(new_models[-1][key], )
        for i in range(len(clients)):
            clients[i].model.load_state_dict(new_models[i])
        print("round {0} complete".format(e+1))
        
        # Run the client with lowest loss against test dataset to examine performance
        if test_loader is not None:
            sloss, saccu = test_DFL(clients, test_loader, debug=False)
            test_loss.append([sloss])
            test_accuracy.append(saccu)
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
K = 10 # Number of clients
B = 10 # Batch size in clients
E = 1 # Epochs per client update
n = len(dataset1) # Number of training samples in total
num_rounds = 200
max_n = n/num_rounds/K

# Create adjacency matrix
AK = np.zeros((K,K))
# For a complete graph:
AK = np.ones((K,K))
for i in range(K):
    AK[i,i] = 0

# Create agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(dataset1, B, E))

DFL_loss_history, DFL_accuracy_history = train_DFL(AK, clients_DFL, num_rounds=num_rounds, dry_run=False, CK=CK, max_n=max_n, 
                             test_loader=test_loader)
test_DFL(clients_DFL, test_loader)





# DFL with localized data
# Divide up the dataset into partitions. Start from uniform partition first. 
K = 15 # Number of clients
B = 32 # Batch size in clients
E = 1 # Epochs per client update
n = len(dataset1) # Number of training samples in total
num_rounds = 100
max_n = n/num_rounds
# Decide how many samples to assign for each client network
nk = [int(n/K)]*K
nk[-1] = n - np.sum(nk[:-1])
# nkc = [0] + np.cumsum(nk)

# Create adjacency matrix
AK = np.zeros((K,K))
# For a complete graph:
AK = np.ones((K,K))
for i in range(K):
    AK[i,i] = 0

# Create agents and assign data to each agent
partitioned_dataset = torch.utils.data.random_split(dataset1, nk)
clients_DFL_Part = []
for i in range(K):
    clients_DFL_Part.append(ClientNet(partitioned_dataset[i], B, E))

DFL_Part_loss_history, DFL_Part_accuracy_history = train_DFL(
    AK, clients_DFL_Part, num_rounds=num_rounds, dry_run=False, CK=CK, max_n=max_n, 
    test_loader=test_loader, avg_weight=0.9)
test_DFL(clients_DFL_Part, test_loader)