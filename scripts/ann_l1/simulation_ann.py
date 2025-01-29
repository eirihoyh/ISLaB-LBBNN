

# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
from flows import PropagateFlow
from torch.optim.lr_scheduler import MultiStepLR
from utils import (create_data_unif,nr_hidden_layers,get_alphas,clean_alpha, 
include_input_from_layer, network_density_reduction,create_layer_name_list,
average_path_length,ece_score_binary)


np.random.seed(1)



# select the device
# note, this experiment runs very quickly on CPU so no point using GPU here 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


TEST_BATCH_SIZE = 8000

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 1600
epochs = 750
l1_lambda = 1.




dep_level = 0.9 ## 0.0, 0.1,0.5, 0.9
non_lin = False


y,x = create_data_unif(n = 80000, dep_level = dep_level, classification=True, non_lin=non_lin)


xtr,ytr = x[0:64000,:],y[0:64000]
xval,yval = x[64000:72000,:],y[64000:72000]
xtest,ytest = x[72000:,:],y[72000:]



dtrain = torch.tensor(np.column_stack((xtr,ytr)), dtype=torch.float32)
dval = torch.tensor(np.column_stack((xval, yval)), dtype=torch.float32)
dtest = torch.tensor(np.column_stack((xtest, ytest)), dtype=torch.float32)

#savedir = '/Users/larsskaaret-lund/Documents/Eirik paper code/data/Simulated_data/linear/' 
#np.savetxt(savedir + str(dep_level)+'X_train.txt',xtr, delimiter=',')
#np.savetxt(savedir +str(dep_level)+'X_test.txt',xtest, delimiter=',')
#np.savetxt(savedir +str(dep_level)+'Y_train.txt',ytr, delimiter=',')
#np.savetxt(savedir +str(dep_level)+'Y_test.txt',ytest, delimiter=',')



TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE


n, p = xtr.shape
dim = 20
HIDDEN_LAYERS = 4
thresh = 0.005

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(p, dim)])
        self.linears.extend([nn.Linear((dim + p), (dim)) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(nn.Linear((dim+p), 1))
        self.loss = nn.BCELoss(reduction='sum')


    def forward(self, x):
       x_input = x.view(-1, p)
       x = torch.sigmoid(self.linears[0](x_input))
       i = 1
       for l in self.linears[1:-1]:
           x = torch.sigmoid(l(torch.cat((x, x_input),1)))
           i += 1

       out = self.linears[i](torch.cat((x, x_input),1))
       return torch.sigmoid(out)
   
    def mpm(self, x):
       x_input = x.view(-1, p)
       w = self.linears[0].weight.clone().detach()
       b = self.linears[0].bias.clone().detach()
       w[abs(w)<thresh] = 0
       x = torch.sigmoid(torch.matmul(x_input,w.T) + b)
       i = 1
       for l in self.linears[1:-1]:
           w_l = l.weight.clone().detach()
           b_l =l.bias.clone().detach()
           w_l[abs(w_l)<thresh] = 0
           x_ = torch.cat((x, x_input),1)
           preact = torch.matmul(x_,w_l.T) + b_l
           x = torch.sigmoid(preact)
           i += 1
        
    
       w_out = self.linears[i].weight.clone().detach()
       w_out[abs(w_out) < thresh] = 0
       b_out = self.linears[i].bias.clone().detach()
       x_out = torch.cat((x, x_input),1)
       preact = torch.matmul(x_out,w_out.T) + b_out
  
       
       return torch.sigmoid(preact)
   
    
   
    
def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        
        _x = dtrain[old_batch: batch_size * batch, 0:p]
        _y = dtrain[old_batch: batch_size * batch, -1]
 
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data)
        target = target.unsqueeze(1).float()
        
        nll = net.loss(outputs, target)
        l1_regularization = 0.
        for param in net.parameters():
            l1_regularization += param.abs().sum()
    
        loss = nll + l1_lambda * l1_regularization
        loss.backward()
        
        optimizer.step()
        pred = outputs.squeeze().detach().cpu().numpy()
        pred = np.round(pred, 0)
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)

    print('loss', loss.item())
    print('accuracy =', np.mean(accs))
    return loss.item()


def test_ensemble(net,data):
    net.eval()
    test_nll = nn.BCELoss(reduction='mean')
    with torch.no_grad():
        data, target = data[:,0:p].to(DEVICE), data[:,-1].to(DEVICE)


        output1 = net(data)
        out2 = net.mpm(data)
   

        target = target.unsqueeze(1).float()
    
        pred1 = np.round(output1,0)
        pred2 = np.round(out2,0)
        
        ens_acc = torch.mean((pred1 == target)* 1.)
        median_acc = torch.mean((pred2 == target)* 1.)
        nll_full = test_nll(target,pred1)
        nll_sparse =  test_nll(target,pred2)
  

        ece_full = ece_score_binary(output1.squeeze(),target.squeeze())
        ece_mpm = ece_score_binary(out2.squeeze(),target.squeeze())
      
        


        

 
    return ens_acc,median_acc,ece_full,ece_mpm,nll_full,nll_sparse





k = 10

accs = np.zeros(k)
meds = np.zeros(k)
alphas = np.zeros((k,p))
used_weights = np.zeros(k)
path_length = np.zeros(k)
max_length = np.zeros(k)
ece = np.zeros(k)
ece_mpm = np.zeros(k)
lik = np.zeros(k)
lik_mpm = np.zeros(k)

for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = NeuralNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.01)
    scheduler = MultiStepLR(optimizer, milestones=[200,400], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        loss = train(net, optimizer)
        scheduler.step()
        
    
    accs[i],meds[i],ece[i],ece_mpm[i],lik[i],lik_mpm[i] = test_ensemble(net,dtest)
    


    
    layer_names = create_layer_name_list(net = net)
    alp = clean_alpha(net, thresh)
    include_inputs = np.array(include_input_from_layer(alp)) * 1
    alphas[i] = include_inputs.max(axis = 0) ## check if variables have been included from any layer
    density, w, tot_weights = network_density_reduction(alp)
    path_length[i] = average_path_length(alp)[0]
    used_weights[i] = w.item()
    max_length[i] = average_path_length(alp)[1].max()

    


np.savetxt('./simulation-results/alphas_ann'+str(dep_level) +str(non_lin) +'.txt',alphas, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/ens_ann'+str(dep_level) +str(non_lin) +'.txt',accs, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/med_ann'+str(dep_level) +str(non_lin) +'.txt',meds, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/used_weights_ann'+str(dep_level) +str(non_lin) +'.txt',used_weights, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/max_depth_ann'+str(dep_level) +str(non_lin) +'.txt',max_length, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/avg_depth_ann'+str(dep_level) +str(non_lin) +'.txt',path_length, delimiter=',',fmt='%s')

np.savetxt('./simulation-results/ece_ann'+str(dep_level) +str(non_lin) +'.txt',ece, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/ece_mpm_ann'+str(dep_level) +str(non_lin) +'.txt',ece_mpm, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/lik_ann'+str(dep_level) +str(non_lin) +'.txt',lik, delimiter=',',fmt='%s')
np.savetxt('./simulation-results/lik_mpm_ann'+str(dep_level) +str(non_lin) +'.txt',lik_mpm, delimiter=',',fmt='%s')


