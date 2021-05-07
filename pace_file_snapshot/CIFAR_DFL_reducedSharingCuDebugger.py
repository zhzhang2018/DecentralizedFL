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
import copy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         ###### self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class ParameterSharer():
    # Handles parameter sharing. Can work with different setups.
    # Note: Assumes the reception graphs are directed. - Otherwise, communication will take longer time and
    # be more costly, and it's harder to find a good topology. 
    def __init__(self, clients, nk, A, R=1.0, share_mode='l', 
                 avg_alg='epsilon', use_rand=False, percentage_shared=1, consensus_threshold=0.05, 
                 loadname='placeholder_load_filename.npy'):
        # External parameters:
        self.clients = clients # List of client references
        self.nk = nk           # Number of samples in each client
        self.A = np.copy(A)    # Adjacency matrix
        self.R = R             # % of neighbors to share each parameter/layer with
        self.mode = share_mode # Decides which parameters are shared to whom
        self.avg_alg = avg_alg # Decides how consensus is run
        self.use_rand = use_rand # Whether each epoch should randomly choose a different set of parameters
        self.percentage_shared = percentage_shared # If = 1, then full model weights are shared.
        self.consensus_threshold = consensus_threshold # Max error percentage tells you when to stop consensus
        
        # Internal aids:
        self.K = len(clients)
        self.numLayer = 5 # Only counts layers with parameters. Note that each layer has weights and biases.
        self.minNumNeighbor = 0 # If positive, then each parameter must hear from at least this value of clients.
        self.keys = [] # List of key strings in the right order
        self.loadname = loadname
        for key in self.clients[0].model.state_dict().keys():
            self.keys.append(key)
#         try:
#             self.share_maps = np.load(self.loadname)
#         except:
#             print("Generating a new share map thing")
        self.generate_segmentations()
    
    def generate_segmentations(self):
        if self.mode == 'l':
            # Share by layers
            self.share_maps = self.split_layers()
            # Other possibilities include dividing each layer into chunks
        elif self.mode == 'f':
            # The smallest sharing unit is individual filters. 
            self.share_maps = self.split_filters()
        elif self.mode == 'c' or self.mode == 'cu':
            # The smallest sharing unit is output channels
            self.share_maps = self.split_channels()
        elif self.mode == 'p':
            pass
#         if not self.use_rand:
#             np.save(self.loadname, self.share_maps)
    
    def split_layers(self):
        # Helper function that splits models by layers, and dictate who gets who.
        # This field stores numLayer different KxK arrays. Each array documents the indices of the clients
        # where each client would receive weights of each layer. 
        # It is NOT an adjacency matrix. The KxK size allow for maximum case storage.
        KSR = -np.ones((self.K, self.numLayer, self.K), dtype=int)
        
        # Decide which layer should be shared - note that layers don't have the same sizes, so this
        # method of segmentation is prone to wrong overall percentages.
        # Right now we do it by randomly choosing a percentage of layers. 
        layer_list = np.random.permutation(np.arange(self.numLayer))[:int(self.numLayer*self.percentage_shared)]
        
        # First, create an adjacency matrix that doens't include any weights, just in case the given Adj has weights.
        A_noweight = np.zeros(self.A.shape)
        A_noweight[ self.A != 0 ] = 1
        Nis = np.sum(A_noweight, axis=1) # number of neighbors for each client; (N,) shape.
        for i in range(self.K):
            nonzero_entry_inds = np.nonzero(self.A[i,:])[0]
            if Nis[i] <= self.minNumNeighbor:
                # In this case, just fill the corresponding slice with non-zero indices and -1s.
                # The nonzero() would return a tuple, thus the [0] attached to the end.
                for j in layer_list:
                    KSR[i,j,0:len(nonzero_entry_inds)] = nonzero_entry_inds
                    KSR[i,j,len(nonzero_entry_inds):] = -np.ones((self.K-len(nonzero_entry_inds),), dtype=int)
            else:
                R = max(self.minNumNeighbor, int(Nis[i] * self.R))
                for j in layer_list:
                    KSR[i,j,:R] = np.random.permutation(nonzero_entry_inds)[:R]
                    KSR[i,j,R:] = -np.ones((self.K-R,),dtype=int)
        return KSR
    
    def split_filters(self):
        # One filter maps to one input and one output channel. 
        # Each bias parameter should be fixed with an onput channel, but it's harder to implement?. 
        # The way we do this is gather all existing filters in the model, and then choose which to use.
        filter_count = []
        filter_maps = [] # A list, each item corresponds to the things inside one layer
        for i in range(self.numLayer):
            filter_count.append( self.clients[0].model.state_dict()[self.keys[2*i]].shape[0] * 
                                 self.clients[0].model.state_dict()[self.keys[2*i]].shape[1] )
            filter_maps.append( -np.ones((self.K, filter_count[-1], self.K, 3), dtype=int) )
            # Things are NOT stored in the order of filter's channels, or in the order of sender indices.
        num_filters = np.sum(filter_count)
        filter_cumcount = np.cumsum(filter_count)
        
        # First, create an adjacency matrix that doens't include any weights, just in case the given Adj has weights.
        A_noweight = np.zeros(self.A.shape)
        A_noweight[ self.A != 0 ] = 1
        Nis = np.sum(A_noweight, axis=1) # number of neighbors for each client; (N,) shape.

        # For each client, pick a set of parameters that it wants to poll.
        # We assume that in the real implementation, this info would be shared before the optimization starts,
        # so that every client knows to send what to whom, and to receive from whom.
        for i in range(self.K):
            filter_maps_fillcount = [0 for j in range(self.numLayer)]
            chosen_filters = np.random.permutation(np.arange(num_filters))[:int(num_filters*self.percentage_shared)]
            # Transcribe the chosen filters into corresponding indices.
            # Side=right for 0-indexing: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
            # Should never generate things on the right of filter_count index.
            filter_inds = np.searchsorted( filter_cumcount, chosen_filters, side='right' )
            for j,find in enumerate(filter_inds):
                filter_maps_inds1 = (chosen_filters[j] - 
                                     filter_cumcount[ find ]) // self.clients[i].model.state_dict()[
                    self.keys[2*find]].shape[1]
                filter_maps_inds2 = (chosen_filters[j] - 
                                     filter_cumcount[ find ]) % self.clients[i].model.state_dict()[
                    self.keys[2*find]].shape[1]
                
                # Pick which clients to poll for those parameters
                nonzero_entry_inds = np.nonzero(self.A[i,:])[0]
                if Nis[i] <= self.minNumNeighbor:
                    R = len(nonzero_entry_inds)
                else:
                    R = max(self.minNumNeighbor, int(Nis[i] * self.R))
                filter_maps[find][i,filter_maps_fillcount[find],:R,0] = np.random.permutation(nonzero_entry_inds)[:R]
                filter_maps[find][i,filter_maps_fillcount[find],:R,1] = np.ones((R,),dtype=int) * filter_maps_inds1
                filter_maps[find][i,filter_maps_fillcount[find],:R,2] = np.ones((R,),dtype=int) * filter_maps_inds2
                filter_maps_fillcount[find] += 1
        
        return filter_maps
    
    def split_channels(self):
        # Split by output channels. Mostly the same as split_filters().
        filter_count = []
        filter_maps = [] # A list, each item corresponds to the things inside one layer
        for i in range(self.numLayer):
            filter_count.append( self.clients[0].model.state_dict()[self.keys[2*i]].shape[0] )
            filter_maps.append( -np.ones((self.K, filter_count[-1], self.K, 2), dtype=int) )
            # Things are NOT stored in the order of filter's channels, or in the order of sender indices.
        num_filters = np.sum(filter_count)
        filter_cumcount = np.cumsum(filter_count)
        
        # First, create an adjacency matrix that doens't include any weights, just in case the given Adj has weights.
        A_noweight = np.zeros(self.A.shape)
        A_noweight[ self.A != 0 ] = 1
        Nis = np.sum(A_noweight, axis=1) # number of neighbors for each client; (N,) shape.

        # For each client, pick a set of parameters that it wants to poll.
        # We assume that in the real implementation, this info would be shared before the optimization starts,
        # so that every client knows to send what to whom, and to receive from whom.
        if self.mode == 'c':
            for i in range(self.K):
                chosen_filters = np.random.permutation(np.arange(num_filters))[:int(num_filters*self.percentage_shared)]
                filter_inds = np.searchsorted( filter_cumcount, chosen_filters, side='right' )
                for j,find in enumerate(filter_inds):
                    filter_maps_ind = (chosen_filters[j] - filter_cumcount[ find ]) 

                    # Pick which clients to poll for those parameters.
                    # R is the number of clients that'll be polled in the future.
                    # The list of clients is stored at filter_maps[<layer's index>][i,<channel's natural order index>,:R,0]
                    # In addition, the channel's index in its layer is stored at the second index at the same slice.
                    nonzero_entry_inds = np.nonzero(self.A[i,:])[0]
                    if Nis[i] <= self.minNumNeighbor:
                        R = len(nonzero_entry_inds)
                    else:
                        R = max(self.minNumNeighbor, int(Nis[i] * self.R))
                    filter_maps[find][i,filter_maps_ind,:R,0] = np.random.permutation(nonzero_entry_inds)[:R]
                    filter_maps[find][i,filter_maps_ind,:R,1] = np.ones((R,),dtype=int) * filter_maps_ind
        elif self.mode == 'cu':
            chosen_filters = np.random.permutation(np.arange(num_filters))[:int(num_filters*self.percentage_shared)]
            filter_inds = np.searchsorted( filter_cumcount, chosen_filters, side='right' )
            for j,find in enumerate(filter_inds):
                filter_maps_ind = (chosen_filters[j] - filter_cumcount[ find ]) 

                # Pick which clients to poll for those parameters.
                # R is the number of clients that'll be polled in the future.
                # The list of clients is stored at filter_maps[<layer's index>][i,<channel's natural order index>,:R,0]
                # In addition, the channel's index in its layer is stored at the second index at the same slice.
                for i in range(self.K):
                    nonzero_entry_inds = np.nonzero(self.A[i,:])[0]
                    R = len(nonzero_entry_inds)
                    filter_maps[find][i,filter_maps_ind,:R,0] = np.random.permutation(nonzero_entry_inds)[:R]
                    filter_maps[find][i,filter_maps_ind,:R,1] = np.ones((R,),dtype=int) * filter_maps_ind        
        return filter_maps
    
    def split_parameters(self):
        # This one directly splits the sharing into individual weights inside every single layer.
        pass
    
    def run_epoch_update(self, max_iter):
        # The parameter update procedure in the training function was migrated below.
        # But first, we need to store some indicator variables
        ccost = np.zeros((K,)) # Stores communication cost for this round
        dists = np.zeros((K,K,max_iter)) # Stores pairwise distances
        distind = 0 # Stores distance per iteration up to this number
        if self.use_rand:
            self.generate_segmentations()
        
        # Next, the tasks are documenting the original distances, and set up distance thresholds.
        if self.mode == 'l':
            original_dists = np.zeros((self.K, self.numLayer))
            for i in range(self.K):
                for k in range(self.numLayer):
                    client_wise_dist = 0
                    client_wise_key_dist = 0
                    key_w = self.keys[2*k]
                    key_b = self.keys[2*k+1]
                    for j in range(self.K):
                        keydist = (
                            np.linalg.norm( self.clients[j].model.state_dict()[key_w] - 
                                            self.clients[i].model.state_dict()[key_w] ) + 
                            np.linalg.norm( self.clients[j].model.state_dict()[key_b] - 
                                            self.clients[i].model.state_dict()[key_b] )
                        )
                        dists[i,j,0] += keydist
                        if j in self.share_maps[i,k,:]:
                            client_wise_key_dist = max(keydist, client_wise_key_dist)
                    original_dists[i,k] = client_wise_key_dist
            dist_threshold = original_dists * self.consensus_threshold
        elif self.mode == 'f':
            # Again, NOT in the order of parameter channel indices
            original_dists = [np.zeros((self.K, fmp.shape[1])) for fmp in self.share_maps]
            for i in range(self.K):
                for k in range(self.numLayer):
                    for r in range(self.share_maps[k].shape[1]):
                        for j in range(self.K):
                            if self.share_maps[k][i,r,j,0] < 0:
                                break
                            else:
                                keydist = np.linalg.norm(
                                    self.clients[
                                        self.share_maps[k][i,r,j,0]
                                                      ].model.state_dict()[self.keys[k*2]][
                                        self.share_maps[k][i,r,j,1], self.share_maps[k][i,r,j,2]
                                    ] - self.clients[i].model.state_dict()[self.keys[k*2]][
                                        self.share_maps[k][i,r,j,1], self.share_maps[k][i,r,j,2] ]
                                )
                                original_dists[k][i,r] = max(original_dists[k][i,r], keydist)
                    # Also document clientwise distances in full
                    key_w = self.keys[2*k]
                    key_b = self.keys[2*k+1]
                    for j in range(self.K):
                        keydist = (
                            np.linalg.norm( self.clients[j].model.state_dict()[key_w] - 
                                            self.clients[i].model.state_dict()[key_w] ) + 
                            np.linalg.norm( self.clients[j].model.state_dict()[key_b] - 
                                            self.clients[i].model.state_dict()[key_b] )
                        )
                        dists[i,j,0] += keydist
            dist_threshold = np.concatenate(original_dists, axis=1) * self.consensus_threshold
        elif self.mode == 'c' or self.mode == 'cu':
            # Again, NOT in the order of parameter channel indices
            original_dists = [np.zeros((self.K, fmp.shape[1])) for fmp in self.share_maps]
            for i in range(self.K):
                for k in range(self.numLayer):
                    for r in range(self.share_maps[k].shape[1]):
                        #if i==0:
                        #    print(i,k, self.share_maps[k][i,r,0,0])
                        if self.mode == 'c':
                            for j in range(self.K):
                                if self.share_maps[k][i,r,j,0] < 0:
                                    break
                                else:
                                    keydist = np.linalg.norm(
                                        self.clients[
                                    self.share_maps[k][i,r,j,0]
                                                ].model.state_dict()[self.keys[k*2]][
                                            self.share_maps[k][i,r,j,1]
                                        ] - self.clients[i].model.state_dict()[self.keys[k*2]][
                                            self.share_maps[k][i,r,j,1] ]
                                    ) + np.linalg.norm(
                                        self.clients[
                                    self.share_maps[k][i,r,j,0]
                                                ].model.state_dict()[self.keys[k*2+1]][
                                            self.share_maps[k][i,r,j,1]
                                        ] - self.clients[i].model.state_dict()[self.keys[k*2+1]][
                                            self.share_maps[k][i,r,j,1] ]
                                    )
                                    original_dists[k][i,r] = max(original_dists[k][i,r], keydist)
                        elif self.mode == 'cu':
                            maxdistind = 0
                            for j in range(self.K):
                                if self.A[i,j] >= 0:
                                    keydist = np.linalg.norm(
                                        self.clients[j].model.state_dict()[self.keys[k*2]][r] - 
                                        self.clients[i].model.state_dict()[self.keys[k*2]][r]
                                    ) + np.linalg.norm(
                                        self.clients[j].model.state_dict()[self.keys[k*2+1]][r] - 
                                        self.clients[i].model.state_dict()[self.keys[k*2+1]][r]
                                    )
#                                    print(i, " receiving updates from ", j, ", about (layer,seg) = ", (k,r))
                                    original_dists[k][i,r] = max(original_dists[k][i,r], keydist)
                                    if keydist ==  original_dists[k][i,r]:
                                        maxdistind = j # print("Max distance at layer ", k, " parameter ", r, "settled between ", (i,j), " at ", original_dists[k][i,r] )
#                            print("Final distance: ", original_dists[k][i,r], ", obtained at ", (i,maxdistind), ", will give threshold = ", original_dists[k][i,r] * self.consensus_threshold)                        
                    # Also document clientwise distances in full
                    key_w = self.keys[2*k]
                    key_b = self.keys[2*k+1]
                    for j in range(self.K):
                        keydist = (
                            np.linalg.norm( self.clients[j].model.state_dict()[key_w] - 
                                            self.clients[i].model.state_dict()[key_w] ) + 
                            np.linalg.norm( self.clients[j].model.state_dict()[key_b] - 
                                            self.clients[i].model.state_dict()[key_b] )
                        )
                        dists[i,j,0] += keydist
            dist_threshold = [od * self.consensus_threshold for od in original_dists]
#            print(dist_threshold)
#             dist_threshold = np.concatenate(original_dists, axis=1) * self.consensus_threshold
        elif self.mode == 'p':
            pass
        
        reached_consensus = False
        for k in range(max_iter):
            distind += 1
            if reached_consensus or distind >= max_iter:
                break
            reached_consensus = True # In cases that only run one iteration, this variable is untouched.
            
            # Set up variables where we do the averaging calculation without disturbing the previous weights
            new_models = []
            for i in range(self.K):
                new_models.append( copy.deepcopy(self.clients[i].model.state_dict()) )

            # Start averaging towards the goal. 
            # If network is undirected, then we use equal neighbor averaging method.
            # If the netwrok is directed, then polling for consensus updates is impractical. We just use weighted sum.
            if self.mode == 'l':
                updated_dists = np.zeros((self.K, self.numLayer))
                for i in range(self.K):
                    for s in range(self.numLayer):
                        key_w = self.keys[2*k]
                        key_b = self.keys[2*k+1]
                        totalweight = self.nk[i]
                        for j in self.share_maps[i,s,:]:
                            if self.share_maps[i,s,j] >= 0:
                                totalweight += self.nk[j]
                            else:
                                break
                        new_models[i][key_w] *= (self.nk[i] / totalweight)
                        new_models[i][key_b] *= (self.nk[i] / totalweight)
                        for j in self.share_maps[i,s,:]:
                            if self.share_maps[i,s,j] >= 0:
                                new_models[i][key_w] += self.clients[j].model.state_dict()[key_w] * (nk[j]/totalweight)
                                new_models[i][key_b] += self.clients[j].model.state_dict()[key_b] * (nk[j]/totalweight)
                                # Accumulate cost for the sender side
                                ccost[j] += torch.numel(self.clients[j].model.state_dict()[key_w])/10000000
                                ccost[j] += torch.numel(self.clients[j].model.state_dict()[key_b])/10000000
                            else:
                                break
                # For mode l, because the graph is undirected and possibly sparse, there is no consensus guarantee.
                # Thus, we don't record the updated distances, and cheat the consensus detection if-else clause
                # by keeping the updated_distances to 0. 

            elif self.mode == 'f':
                updated_dists = [np.zeros((self.K, fmp.shape[1])) for fmp in self.share_maps]
                for i in range(self.K):
                    for k in range(self.numLayer):
                        for r in range(self.share_maps[k].shape[1]):
                            if self.share_maps[k][i,r,0,0] < 0:
                                break
                            # Gather the weights during consensus
                            totalweight = self.nk[i]
                            for j in self.share_maps[k][i,r,:,0]:
                                if self.share_maps[k][i,r,j,0] < 0:
                                    break
                                totalweight += self.nk[j]
                            # Sum up the parameters and give them weights
                            new_models[i][self.keys[2*k]] *= (self.nk[i] / totalweight)
#                             new_models[i][key_b] *= (self.nk[i] / totalweight)
                            for j in self.share_maps[k][i,r,:,0]:
                                if self.share_maps[k][i,r,j,0] < 0:
                                    break
                                # The huge mess in brackets would be the weights' indices
                                new_models[i][self.keys[k*2]][
                                            self.share_maps[k][i,r,j,1], self.share_maps[k][i,r,j,2]
                                        ] += self.clients[
                                                self.share_maps[k][i,r,j,0]
                                                         ].model.state_dict()[self.keys[k*2]][
                                            self.share_maps[k][i,r,j,1], self.share_maps[k][i,r,j,2]
                                        ] * (nk[j]/totalweight)
#                                 new_models[i][key_b] += self.clients[j].model.state_dict()[key_b] * (nk[j]/totalweight)
                                # Accumulate cost for the sender side
                                ccost[self.share_maps[k][i,r,j,0]] += torch.numel(
                                    self.clients[self.share_maps[k][i,r,j,0]].model.state_dict()[self.keys[k*2]][
                                        self.share_maps[k][i,r,j,1], self.share_maps[k][i,r,j,2]
                                            ])/10000000
                updated_dists = np.concatenate(updated_dists, axis=1)
            elif self.mode == 'c':
                updated_dists = [np.zeros((self.K, fmp.shape[1])) for fmp in self.share_maps]
                for i in range(self.K):
                    for k in range(self.numLayer):
                        for r in range(self.share_maps[k].shape[1]):
                            if self.share_maps[k][i,r,0,0] < 0:
                                break
                            # Gather the weights during consensus
                            totalweight = self.nk[i]
                            for j in self.share_maps[k][i,r,:,0]:
                                if self.share_maps[k][i,r,j,0] < 0:
                                    break
                                totalweight += self.nk[j]
                            # Sum up the parameters and give them weights
                            new_models[i][self.keys[2*k  ]] *= (self.nk[i] / totalweight)
                            new_models[i][self.keys[2*k+1]] *= (self.nk[i] / totalweight)
                            for j in self.share_maps[k][i,r,:,0]:
                                if self.share_maps[k][i,r,j,0] < 0:
                                    break
                                # The huge mess in brackets would be the weights' indices
                                new_models[i][self.keys[k*2]][self.share_maps[k][i,r,j,1]] += self.clients[
                                        self.share_maps[k][i,r,j,0]
                                                ].model.state_dict()[self.keys[k*2]][
                                                    self.share_maps[k][i,r,j,1]
                                                        ] * (nk[j]/totalweight)
                                new_models[i][self.keys[k*2+1]][self.share_maps[k][i,r,j,1]] += self.clients[
                                        self.share_maps[k][i,r,j,0]
                                                ].model.state_dict()[self.keys[k*2+1]][
                                                    self.share_maps[k][i,r,j,1]
                                                        ] * (nk[j]/totalweight)
                                
                                # Accumulate cost for the sender side
                                ccost[self.share_maps[k][i,r,j,0]] += (                   torch.numel(
                                    self.clients[self.share_maps[k][i,r,j,0]].model.state_dict()[
                                        self.keys[k*2  ]][self.share_maps[k][i,r,j,1]]) + torch.numel(
                                    self.clients[self.share_maps[k][i,r,j,0]].model.state_dict()[
                                        self.keys[k*2+1]][self.share_maps[k][i,r,j,1]])              )/10000000
                updated_dists = np.concatenate(updated_dists, axis=1)
            elif self.mode == 'cu':
                updated_dists = [np.zeros((self.K, fmp.shape[1])) for fmp in self.share_maps]
                if avg_alg == 'epsilon':
                    Ni = np.max( np.sum(self.A, axis=0) )+1
                    print(0.99/Ni, Ni)
                    avg_weight = max(0.99/Ni, 0.001)
                    for i in range(self.K):
                        for j in range(self.K):
                            if self.A[i,j] > 0:
                                for ki in range(self.numLayer):
                                    for r in range(self.share_maps[ki].shape[1]):
                                        if self.share_maps[ki][i,r,0,0] < 0:
                                            continue
                                        new_models[i][self.keys[2*ki  ]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[ki*2]][r] - 
                                            self.clients[i].model.state_dict()[self.keys[ki*2]][r] )
                                        new_models[i][self.keys[2*ki+1]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[ki*2+1]][r] - 
                                            self.clients[i].model.state_dict()[self.keys[ki*2+1]][r] )
                                        # Accumulate cost for the sender side
                                        ccost[j] += (torch.numel( self.clients[j].model.state_dict()[self.keys[ki*2]][r] ) + 
                                                     torch.numel( self.clients[j].model.state_dict()[self.keys[ki*2+1]][r])
                                                    )/10000000
#                                        print(i, " receiving updates from ", j, ", about (layer,seg) = ", (ki,r))
#                                        if r == 0 and i==0 and ki < 2:
#                                            print("Comparing client {0} and {1} at layer {2} parameter {3}: ".format(i,j,ki,r), 
# new_models[i][self.keys[2*ki  ]][r,0,0,:], " vs ", new_models[j][self.keys[2*ki  ]][r,0,0,:] )
#                                    if ki == 0 and i==0:
#                                        print('After share: ', torch.sum(new_models[i][self.keys[2*ki+1]][r]), torch.sum(new_models[i][self.keys[2*ki]][r]) )                        
                elif avg_alg == 'metropolis':
                    Nis = np.sum(self.A, axis=0)
                    avg_weight = 1 #0.99/Ni
                    for i in range(self.K):
                        for k in range(self.numLayer):
                            for r in range(self.share_maps[k].shape[1]):
                                if self.share_maps[k][i,r,0,0] < 0:
                                    break
                                for j in range(self.K):
                                    if self.A[i,j] > 0:
                                        new_models[i][self.keys[2*k  ]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[k*2]][r] - 
                                            self.clients[i].model.state_dict()[self.keys[k*2]][r] ) / max(Nis[i], Nis[j], 0.001)
                                        new_models[i][self.keys[2*k+1]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[k*2+1]][r] - 
                                            self.clients[i].model.state_dict()[self.keys[k*2+1]][r] ) / max(Nis[i], Nis[j], 0.001)
                                        # Accumulate cost for the sender side
                                        ccost[j] += (torch.numel( self.clients[j].model.state_dict()[self.keys[k*2]][r] ) + 
                                                     torch.numel( self.clients[j].model.state_dict()[self.keys[k*2+1]][r])
                                                    )/10000000
                else:
                    for i in range(self.K):
                        Ni = np.sum(self.A[i,:])
                        for k in range(self.numLayer):
                            for r in range(self.share_maps[k].shape[1]):
                                if self.share_maps[k][i,r,0,0] < 0:
                                    break
                                for j in range(self.K):
                                    if self.A[i,j] > 0:
                                        new_models[i][self.keys[2*k  ]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[k*2]][r] )
                                        new_models[i][self.keys[2*k+1]][r] += avg_weight * self.A[i,j] * (
                                            self.clients[j].model.state_dict()[self.keys[k*2+1]][r] )
                                        # Accumulate cost for the sender side
                                        ccost[j] += (torch.numel( self.clients[j].model.state_dict()[self.keys[k*2]][r] ) + 
                                                     torch.numel( self.clients[j].model.state_dict()[self.keys[k*2+1]][r])
                                                    )/10000000
                        for key in new_models[i].keys():
                            new_models[i][key] /= (Ni+1)
                # Document updated distances
                for i in range(self.K):
                    for ki in range(self.numLayer):
                        for r in range(self.share_maps[ki].shape[1]):
                            if self.share_maps[ki][i,r,0,0] < 0:
                                continue
                            for j in range(self.K):
                                if self.A[i,j] >= 0:
                                    keydist = np.linalg.norm(
new_models[j][self.keys[2*ki  ]][r]-new_models[i][self.keys[2*ki  ]][r]
                                    ) + np.linalg.norm(
new_models[j][self.keys[2*ki+1]][r]-new_models[i][self.keys[2*ki+1]][r]
                                    )
                                    if keydist > dist_threshold[ki][i,r]:
                                        print("Failed consensus at layer ", ki, ", channel ", r, ", between agents ", (i,j), " at iteration ", k)
                                        print("Current distance = ", keydist, " while goal distance = ", dist_threshold[ki][i,r])
                                        reached_consensus = False
                                        break
                            if not reached_consensus:
                                break
                        if not reached_consensus:
                            break
                    if not reached_consensus:
                        break
#                                     updated_dists[k][i,r] += keydist
#                                     updated_dists[k][i,r] = max(updated_dists[k][i,r], keydist)
#                 updated_dists = np.concatenate(updated_dists, axis=1)
            elif self.mode == 'p':
                if avg_alg == 'epsilon':
                    Ni = np.min( np.sum(adj, axis=0) )
                    avg_weight = min(avg_weight, 0.99/Ni)
                    for i in range(len(clients)):
                        for j in range(len(clients)):
                            for key in new_models[j].keys():
                                new_models[i][key] += avg_weight * adj[i,j] * (clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key])
                                ccost[i] += torch.numel(clients[j].model.state_dict()[key])/10000000
                                ccost[j] += torch.numel(clients[j].model.state_dict()[key])/10000000
                elif avg_alg == 'metropolis':
                    Nis = np.sum(adj, axis=0)
                    for i in range(len(clients)):
                        for j in range(len(clients)):
                            for key in new_models[j].keys():
                                new_models[i][key] += avg_weight * adj[i,j] * (clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key]) / max(Nis[i], Nis[j])
                                ccost[i] += torch.numel(clients[j].model.state_dict()[key])/10000000
                                ccost[j] += torch.numel(clients[j].model.state_dict()[key])/10000000
                else:
                    for i in range(len(clients)):
                        Ni = np.sum(adj[i,:])
                        # Calcualte the averages, or use other methods
                        for j in range(len(clients)):
                            for key in new_models[j].keys():
                                new_models[i][key] += avg_weight * adj[i,j] * clients[j].model.state_dict()[key]
                                ccost[i] += torch.numel(clients[j].model.state_dict()[key])/10000000
                                ccost[j] += torch.numel(clients[j].model.state_dict()[key])/10000000
                        for key in new_models[i].keys():
    #                         new_models[i][key] -= clients[i].model.state_dict()[key]
                            new_models[i][key] /= (Ni+1) # Or use torch.div()

            # Load averaged results of this iteration
            for i in range(self.K):
                self.clients[i].model.load_state_dict(new_models[i])
            
            # Record distance data
            for i in range(self.K):
                for j in range(self.K):
                    client_wise_dist = 0
                    if j>i: # Avoid double checking
                        for key in self.keys:
                            client_wise_dist += np.linalg.norm( 
                                self.clients[j].model.state_dict()[key] - 
                                self.clients[i].model.state_dict()[key] )
                        dists[i,j,distind] = client_wise_dist
                        dists[j,i,distind] = client_wise_dist
            
            # Check the distances and decide if the consensus can stop (Moved to the inside of loops instead)
#             reached_consensus = np.all( updated_dists <= dist_threshold ) 
            print("Currently at: Iteration ", k, "\n\n")
            
            if reached_consensus:
                break
            
        print(dists[0,8,:])
        if k < max_iter-1:
            print("Finished averaging at the {}-th iteration".format(k))
        else:
            print("Probably failed to converge in consensus lol")
        # Return things the training function should need to know
        return ccost, dists[:,:,:distind+1], distind
            
# Parameters for training
batch_size = 64
test_batch_size = 1000
no_cuda = False
dry_run = False
seed = 1

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

# transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
#     ])
# dataset1 = datasets.MNIST('../data', train=True, download=False,#True,
#                    transform=transform)
# dataset2 = datasets.MNIST('../data', train=False,
#                    transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
# test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
dataset1 = datasets.CIFAR10('../data/cifar-10-batches-py', train=True, download=False,#True,
                   transform=transform)
dataset2 = datasets.CIFAR10('../data/cifar-10-batches-py', train=False,
                   transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Implement client logic
class ClientNet():
    def __init__(self, dataset, B=16, E=1, optimizerlr=1, schedgamma=0.7, optim_method=optim.Adadelta, loss_f=F.nll_loss):
        self.model = Net().to(device)
        self.optimizer = optim_method(self.model.parameters(), lr=optimizerlr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=schedgamma)
        self.loss_f = loss_f
        self.loss_history = []
        self.B = B
        self.E = E
        train_kwargs = {'batch_size': B, 'shuffle': True}
        self.train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)
        
    def update(self, dry_run=False, max_n=np.inf, weights=None):
        if weights is not None:
            self.model.load_state_dict(weights)
        data_sample_count = 0
        self.model.train()
        for e in range(self.E):
            self.loss_history.append([])
            # How to only load a limited amount of data?
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_f(output, target)
                loss.backward()
                self.optimizer.step()

                self.loss_history[-1].append(loss.item())
                data_sample_count += len(data)
                if dry_run or batch_idx*self.B >= max_n:
                    # batch_idx grows 1 at a time. 
                    # The loop will be over when all data is used, and batch_idx*self.B might be larger than that.
                    break
        return data_sample_count
        
# Try to see if you can train distributedly - i.e. average between several separate models
# Implement train and test logic. This one doens't have a server, and I assume we can let them all update together.
# dists: expected to be a K x K x num_avg_iter x num_rounds array to store distances
def train_DFL(adj, clients, dists, comm_costs, samples_seen, sharer,
              num_rounds=1, dry_run=False, debug=False, CK=10, max_n=np.inf, test_loader=None,
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
        for i,m in enumerate(clients):
            samples_seen_by_m = m.update(dry_run, max_n=max_n)
            samples_seen[i,e] = samples_seen_by_m
        # Aggregation for each client. How it's done depends on the num_avg_iter argument:
        # num_avg_iter > 0: Execute at most this number of consensus averaging.
        # num_avg_iter < 0: Execute until the system reaches consensus (with pairwise errors smaller than avg_error_thres)
        print("Finished updating for {}-th round".format(e))
        
        ccost, kdist, kdistind = sharer.run_epoch_update(dists.shape[2])
        
        # Fill the rest of the dists array with the last distance info after breaking out
        dists[:,:,:kdistind+1,e] = kdist
        for k in range(kdistind+1, dists.shape[2]):
            dists[:,:,k,e] = dists[:,:,kdistind,e]
        
        # Record communication cost
        comm_costs[:,e] = ccost
        
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
                test_accuracy.append(saccus)
#                 test_accuracy.append(np.max(saccu))
            else:
                # Only record the best client's performance
                sloss, saccu = test_DFL(clients, test_loader, debug=False)
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
            
        print("round {0} complete with {1} averaging steps and test accuracy {2}".format(e+1,kdistind,test_accuracy[-1]))
        
        # Preemptively save the parameters and the clients
        if bother_saving > 0 and ((e+1) % bother_saving == 0):
            with open(save_model_to, "wb") as fp:
                pickle.dump([test_loss, test_accuracy, e], fp)
            for i,c in enumerate(clients):
                torch.save(c.model.state_dict(), save_model_to+'_client{}'.format(i))
                with open(save_model_to+'_client{}_loss_history'.format(i), "wb") as fp:
                    pickle.dump(c.loss_history, fp)
            np.save(script_filename+file_identifier+'_dists_history_temp.npy', dists)
            np.save(script_filename+file_identifier+'_communication_cost_history.npy', comm_costs)
            np.save(script_filename+file_identifier+'_sample_usage.npy', samples_seen)
        
    return test_loss, test_accuracy
    
def test_DFL(clients, test_loader, debug=True):
    # Pick the client with the lowest loss to run against test dataset
    losses = []
    for c in clients:
        c.model.eval()
#         print(c.loss_history)
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
# This file allows custom arguments for the file. First attempt.
import argparse # https://docs.python.org/3/library/argparse.html#module-argparse
parser = argparse.ArgumentParser() # https://docs.python.org/3/howto/argparse.html
# parser.add_argument("var_to_be_recorded", type=int, # Calls the variable at this place as "var_to_be_recorded"
#                     help="records the given value") # This one is order-sensitive and must be specified.
# parser.add_argument( "-v", "--verbose", help="increase output verbosity", choices=[0, 1, 2],
#                     action="store_true" ) # Note: Can't have both "choices" and "action" in real lines.
parser.add_argument( "file_identifier", help="Unique identifier for experiment data files", 
                    default='' ) # https://stackoverflow.com/a/15301183
parser.add_argument( "file_type", help="Specifies the constitution of client types", 
                    default='A' )
parser.add_argument( "-K", "--num_clients", help="Number of clients", type=int, default=10 )
parser.add_argument( "-k", "--num_neighbors", help="Average number of neighbors per client", type=int, default=2 )
parser.add_argument( "-p", "--rewire_probs", help="Probability of rewiring in small-world", type=float, default=0.5 )
parser.add_argument( "-C", "--update_percentage", help="Percentage of client that updates in each epoch", 
                    type=float, default=1 )
parser.add_argument( "-B", "--batch_size", help="Size of minibatch during update", type=int, default=64 )
parser.add_argument( "-E", "--epoch_per_update", help="Number of epochs per client update", type=int, default=1 )
parser.add_argument( "-nr", "--num_rounds", help="Number of epochs in this training", type=int, default=200 )
parser.add_argument( "-EE", "--epoch_data_portion", help="How many times of total local data are used per epoch", 
                    type=float, default=0.05 )
parser.add_argument( "-s", "--overlap_percentage", help="How much overlap in training data", 
                    type=float, default=0 )
parser.add_argument( "-Mean", "--partition_mean", help="Mean size if partitioning the dataset at random", 
                    type=float, default=1 )
parser.add_argument( "-V", "--partition_variance", help="Variance if partitioning the dataset at random", 
                    type=float, default=0.0 )
parser.add_argument( "-S", "--SameInit", help="Set same initial values among clients", action="store_true")
parser.add_argument( "-U", "--use_unbalanced_dataset", help="Use unbalanced dataset", action="store_true")
parser.add_argument( "-D", "--dry_run", help="Only try to train the first batch", action="store_true")
parser.add_argument( "-AAL", "--avg_alg", help="Consensus averaging algorithm", default='epsilon' )
parser.add_argument( "-AET", "--avg_error_thres", help="Error limit during consensus", 
                    type=float, default=0.1 )
parser.add_argument( "-Adj", "--adjacency_matrix", help="Network topology", default='SW' )
parser.add_argument( "-R", "--neighbor_percentage", 
                     help="Percentage of neighbors that would randomly be chosen to share a parameter / a block of parameter", type=float, default=1.0 )
parser.add_argument( "-SM", "--share_mode", 
                     help="Determines how parameters are segmented and shared.", default='l' )
parser.add_argument( "-URS", "--use_rand_segmentation", 
                     help="If segmentation should change every epoch", action="store_true" )
parser.add_argument( "-PSS", "--percentage_shared_segmentation", 
                     help="Percentage of parameters being shared", type=float, default=1.0 )
parser.add_argument( "-CT", "--consensus_threshold", 
                     help="Threshold (percentage) after which consensus can stop", type=float, default=0.05 )

args = parser.parse_args()
file_identifier = args.file_identifier
file_type = args.file_type
K = args.num_clients
k = args.num_neighbors
p = args.rewire_probs
C = args.update_percentage
B = args.batch_size
E = args.epoch_per_update
num_rounds = args.num_rounds
EE = args.epoch_data_portion
S = args.overlap_percentage
Mean = args.partition_mean
V = args.partition_variance
SameInit = args.SameInit
use_unbalanced_dataset = args.use_unbalanced_dataset
dry_run = args.dry_run
avg_alg = args.avg_alg
avg_error_thres = args.avg_error_thres
adjacency_matrix = args.adjacency_matrix
# Sharing parameter parameters
R = args.neighbor_percentage
share_mode = args.share_mode
use_rand_segmentation = args.use_rand_segmentation
pss = args.percentage_shared_segmentation
consensus_threshold = args.consensus_threshold

n = len(dataset1) # Number of training samples in total
max_n = n/K * EE

import utils
nk = utils.generate_partition_sizes(K, n, mean=Mean, variance=V, options=None, minsize=B)

# Create small world adjacency matrix
np.random.seed(202)
if adjacency_matrix == 'SW':
    AC = np.zeros((K,K))
    for i in range(1,k//2+1):
        for j in range(K):
            AC[j, (j+i)%K] = 1
            AC[(j+i)%K, j] = 1
    Adj = np.copy(AC)
    
    # Go through all the edges
    for i in range(1,K):
        for j in range(i,K):
            if AC[i,j] == 1 and np.random.random() < p:
                # Pick a random node as the new neighbor
                i_add = np.random.randint(1,K)
                if AC[i,(j+i_add)%K] == 0 and i != (j+i_add)%K:
#                     print("Rewiring {} to {}".format((i,j), (i,(j+i_add)%K)))
                    Adj[i,j] = 0
                    Adj[j,i] = 0
                    Adj[i,(j+i_add)%K] = 1
                    Adj[(j+i_add)%K,i] = 1
elif adjacency_matrix == 'AK' or adjacency_matrix == 'complete':
    AK = np.ones((K,K))
    for i in range(K):
        AK[i,i] = 0
    Adj = AK
elif adjacency_matrix == 'AC' or adjacency_matrix == 'cycle':
    AC = np.zeros((K,K))
    AC[0,-1] = 1
    AC[-1,0] = 1
    for i in range(K-1):
        AC[i,i+1] = 1
        AC[i+1,i] = 1
    Adj = AC
elif adjacency_matrix == 'ACk' or adjacency_matrix == 'cycle_k':
    AC = np.zeros((K,K))
    for i in range(1,k//2+1):
        for j in range(K):
            AC[j, (j+i)%K] = 1
            AC[(j+i)%K, j] = 1
    Adj = AC
print(np.linalg.eig(np.diag(np.sum(Adj,axis=0)) - Adj))
print(Adj)
np.random.seed(seed)

loss_fs = [F.nll_loss] * K
list_size_counts = [K]

filetype_dict = {'A':optim.Adam ,'B':optim.Adadelta, 'C':optim.Adagrad, 'D':optim.RMSprop, 'E':optim.SGD}
filetype_lr_dict = {'A':0.001, 'B':0.01, 'C':0.005, 'D':0.0005, 'E':0.05}

if file_type[0] == 'F':
    if file_type[-1] == 'm':
        list_size_counts = [K-1, 1]
        file_type = file_type[:-1]
    elif file_type[-1] == 'n':
        list_size_counts = [K-3, 3]
        file_type = file_type[:-1]
    elif file_type[-1] == 's':
        list_size_counts = [K-(K//2), K//2]
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
    elif file_type == 'Ac15':
        lr_list = 0.001 + 0.0005 * np.arange(1,K+1) # Bad; 98% when starting uniform
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
    if file_type == 'E1':
        lr_list = [0.1] * K # Naughtily sensitive
    elif file_type == 'E2':
        lr_list = [0.01] * K # Naughtily sensitive
    elif file_type == 'Ec2':
        lr_list = 0.01 * np.arange(1,K+1) # Noisy 95.5%
    elif file_type == 'Ec12':
        lr_list = 0.01 * np.array([1,3,5,7,9,20,40,60,80,100]) # 96.83% Very noisy
    optim_methods = [optim.SGD] * K
# F stands for Fusion
elif file_type[0] == 'F':
    if file_type == 'F_AC': 
        lr_list = [0.001] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_AD': 
        lr_list = [0.001] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
    elif file_type == 'F_AE': 
        lr_list = [0.001] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_AB': 
        lr_list = [0.001] * list_size_counts[0] + [0.01] * list_size_counts[1] 
        optim_methods = [optim.Adam] * list_size_counts[0] + [optim.Adadelta] * list_size_counts[1] 
    elif file_type == 'F_CA': 
        lr_list = [0.005] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_CD': 
        lr_list = [0.005] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
    elif file_type == 'F_CE': 
        lr_list = [0.005] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        optim_methods = [optim.Adagrad] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_DA': 
        lr_list = [0.0005] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_DC': 
        lr_list = [0.0005] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_DE': 
        lr_list = [0.0005] * list_size_counts[0] + [0.05] * list_size_counts[1] 
        optim_methods = [optim.RMSprop] * list_size_counts[0] + [optim.SGD] * list_size_counts[1] 
    elif file_type == 'F_EA': 
        lr_list = [0.05] * list_size_counts[0] + [0.001] * list_size_counts[1] 
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adam] * list_size_counts[1] 
    elif file_type == 'F_EB': 
        lr_list = [0.05] * list_size_counts[0] + [0.01] * list_size_counts[1] 
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adadelta] * list_size_counts[1] 
    elif file_type == 'F_EC': 
        lr_list = [0.05] * list_size_counts[0] + [0.005] * list_size_counts[1] 
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.Adagrad] * list_size_counts[1] 
    elif file_type == 'F_ED': 
        lr_list = [0.05] * list_size_counts[0] + [0.0005] * list_size_counts[1] 
        optim_methods = [optim.SGD] * list_size_counts[0] + [optim.RMSprop] * list_size_counts[1] 
# G stands for Group Fusion
elif file_type[0] == 'G':
    # Expects some code of the form 'G_<Tx>' where <> can be repeated in whatever order, T=type, x=count
    filetypes = file_type[2:]
    lr_list = []
    optim_methods = []
    strind = 0
    while strind < len(filetypes):
        if filetypes[strind] >= 'A' and filetypes[strind] <= 'z':
            optim_method = filetype_dict[filetypes[strind]]
            lr_val = filetype_lr_dict[filetypes[strind]]
            strind += 1
        else:
            numfin = strind
            while numfin < len(filetypes) and filetypes[numfin] <= '9' and filetypes[numfin] >= '0':
                numfin += 1
            optim_count = int(filetypes[strind:numfin])
            optim_methods += [optim_method]*optim_count
            lr_list += [lr_val]*optim_count
            strind = numfin
    if len(lr_list) < K:
        optim_count = K - len(lr_list)
        optim_methods += [optim_method]*optim_count
        lr_list += [lr_val]*optim_count
        
else:
    file_type = 'BassLine'
    lr_list = [1] * K
    optim_methods = [optim.Adadelta] * K
    
# Attach "SameInit" to the end of file_type if specified
if SameInit:
    file_type += 'SameInit'

unb_name = ''
if use_unbalanced_dataset:
    unb_name = 'Unbalanced'

# script_filiename = 'MNIST_DFL_reduceCom{}_{}dataset_K{}_C{}_E{}_S{}_M{}_V{}_'.format(file_type,unb_name,K,C,E*EE,S,Mean,V)
script_filiename = 'CIFAR_DFL_reduceCom{}_{}dataset_'.format(file_type,unb_name) + file_identifier + '_'.join(sys.argv[2:])
# script_filename = 'tmp_data/'+script_filiename
script_filename = '../../../../../../scratch1/7/zzhang433/tmp_data/'+script_filiename
# Prepare some files for backup purposes.
# Note: If we want to continue where we left off, then it's reasonable to assume that we know what's the settings
# from the previous run. In other words, we assume that the variables above of this section are still the same. 
# Thus, if you try to load things with different settings, expect unexpected behaviors.
    
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
    '''
    nk = [len(dts) for dts in datasets_by_label]
    nshare = [int(nsample / (K-1) * S) for nsample in nk]
    partitioned_full_dataset = datasets_by_label
    print("Had to modify the dataset partition to create unbalanced dataset. But still, it is done.")
    print(nk)
    '''
    # An alternate way is to randomly decide how much data each client is going to get, and then
    # assign classes in the order from 0 to 9 (or shuffle them). One way to make sure the assignment always 
    # gives a sensible outcome is to pre-determine the percentage of data from each class for each client.
    # For example, pre-determine a percentage of class-0 data that would be assigned to the clients where this
    # class will dominate, and then randomly partition the rest to the other clients. 
    # Depending on K, the number of clients that receive each class as dominated data would differ between classes.
    # Hopefully this can be overcome by some careful modifications on the randomness.
    class_share_percentage = 0.1
    class_assign_ind = np.random.randint(0,10, (K,) )
    
    # Check if all classes have got at least a dominant client
    all_class_dominate_once = False
    while (not all_class_dominate_once):
        for i in range(10):
            if i not in class_assign_ind:
                class_assign_ind[np.random.randint(0,K)] = i
        all_class_dominate_once = True
        for i in range(10):
            if i not in class_assign_ind:
                all_class_dominate_once = False
    # Count those clients
    class_dominate_counts = []
    for i in range(10):
        class_dominate_counts.append( len([j for j in class_assign_ind if j == i]) )
    # Assign percentages
    class_client_sample_count = np.zeros((10,K))
    for i in range(10):
        # Make random values and normalize to create a distribution
        rand_for_class = np.random.uniform(size=(K-class_dominate_counts[i],))
        rand_for_class = class_share_percentage * rand_for_class / np.sum(rand_for_class)
        # Insert dominant ones into the array
        for j in range(K):
            if class_assign_ind[j] == i:
                rand_for_class = np.insert(rand_for_class, j, (1-class_share_percentage)/class_dominate_counts[i])
        class_client_sample_count[i] = rand_for_class * len(datasets_by_label[i])
    # Turn percentages into counts
    class_client_sample_count = [ [int(c.item()) for c in class_client_sample_count[i]] for i in range(10)]
    # Check if the counts would match
    for i in range(10):
        class_client_sample_count[i][-1] -= (sum(class_client_sample_count[i]) - len(datasets_by_label[i]))
    # Assign the samples based on counts
    partitioned_full_dataset = []
    partitioned_full_clientwise_dataset = []
    for i in range(10):
        partitioned_full_clientwise_dataset.append( torch.utils.data.random_split(datasets_by_label[i], class_client_sample_count[i]) )
    for i in range(10):
        for j in range(K):
            if i == 0:
                partitioned_full_dataset.append( partitioned_full_clientwise_dataset[i][j] )
            else:
                partitioned_full_dataset[j] += partitioned_full_clientwise_dataset[i][j]
    nk = [len(dts) for dts in partitioned_full_dataset]
    nshare = [int(nsample / (K-1) * S) for nsample in nk]
    print(nk)
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
torch.manual_seed(seed) # Return to previous seed

Es = [E]*K # baseline (b

# Create agents
clients_DFL = []
for i in range(K):
    clients_DFL.append(ClientNet(partitioned_dataset[i], B, E=Es[i], optimizerlr=lr_list[i], 
                                 optim_method=optim_methods[i], loss_f=loss_fs[i]))

sharer = ParameterSharer(clients_DFL, nk, Adj, R=R, share_mode=share_mode, avg_alg=avg_alg, 
                         use_rand=use_rand_segmentation, percentage_shared=pss, 
                         consensus_threshold=consensus_threshold,
                         loadname=script_filename+file_identifier+'_communication_sharemap.npy')

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

try:
    comm_costs = np.load(script_filename+file_identifier+'_communication_cost_history.npy')
except:
    print("Did not find cost history named '"+script_filename+file_identifier+"_communication_cost_history.npy'. Starting from zeros.")
    comm_costs = np.zeros((K,num_rounds))

try:
    samples_seen = np.load(script_filename+file_identifier+'_sample_usage.npy')
except:
    print("Did not find sample counts named '"+script_filename+file_identifier+"_sample_usage.npy'. Starting from 0.")
    samples_seen = np.zeros((K,num_rounds))

DFL_loss_history, DFL_accuracy_history = train_DFL(Adj, clients_DFL, dists, comm_costs, samples_seen, sharer, 
                                    num_rounds=num_rounds, dry_run=dry_run, 
                                                   CK=K, max_n=max_n, 
                             test_loader=test_loader, test_all_client=True,
              num_avg_iter=num_avg_iter, max_num_avg_iter=100, avg_error_thres=avg_error_thres,
              past_params=params, save_model_to=script_filename+file_identifier, bother_saving=min(1,int(200/E/EE)), #10, 
                                                   avg_alg=avg_alg, avg_weight=0.05)
test_DFL(clients_DFL, test_loader)

# Modify the save filename based on properties
# script_filiename = script_filiename + file_identifier + '_'.join(sys.argv[2:])
for i in range(K):
    np.save(script_filiename+'client_{0}_loss_history'.format(i), clients_DFL[i].loss_history)
np.save(script_filiename+'Accuracy', DFL_accuracy_history)
np.save(script_filiename+'average_loss_history',DFL_loss_history)
np.save(script_filiename+'dists_history',dists[:,:,[0,-1],:])
np.save(script_filiename+'comm_costs_history',comm_costs)
np.save(script_filiename+'sample_count_history',samples_seen)
