# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchmetrics import R2Score

import pandas as pd
from flows import PropagateFlow
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from utils import (nr_hidden_layers,get_alphas,clean_alpha, 
include_input_from_layer, network_density_reduction,create_layer_name_list,
average_path_length,mean_pinball_loss)






from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  


np.random.seed(1)




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'
TEST_SAMPLES = 10
TEST_BATCH_SIZE = 418

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 752
epochs = 2000
thresh = 0.005
dim = 200
HIDDEN_LAYERS = 2

df = abalone.data
X_original = abalone.data.features
X_original = np.array(pd.get_dummies(X_original, drop_first=True)*1.)
y_original = np.array(abalone.data.targets.squeeze(),dtype = np.float32)


X, X_test, y, y_test = train_test_split(X_original, y_original, test_size=0.10, random_state=42)

#savedir = '/Users/larsskaaret-lund/Documents/Eirik paper code/data/abalone'
#np.savetxt(savedir + '/X_train.txt',X, delimiter=',')
#np.savetxt(savedir + '/X_test.txt',X_test, delimiter=',')
#np.savetxt(savedir + '/Y_train.txt',y, delimiter=',')
#np.savetxt(savedir + '/Y_test.txt',y_test, delimiter=',')






dtest = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
dtrain = torch.tensor(np.column_stack((X,y)),dtype = torch.float32)


TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE


n, p = X.shape

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(p, dim)])
        self.linears.extend([nn.Linear((dim + p), (dim)) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(nn.Linear((dim+p), 1))
        self.loss = nn.MSELoss(reduction='sum')


    def forward(self, x):
       x_input = x.view(-1, p)
       x = torch.sigmoid(self.linears[0](x_input))
       i = 1
       for l in self.linears[1:-1]:
           x = torch.sigmoid(l(torch.cat((x, x_input),1)))
           i += 1

       out = self.linears[i](torch.cat((x, x_input),1))
       return out
   
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
  
       
       return preact
   
    
   
    
def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
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
    
        loss = nll + l1_regularization
        

        loss.backward()
        optimizer.step()
        

    print('loss', loss.item())
 
    return loss.item()


def test_ensemble(net,data):
    net.eval()
    crit = nn.MSELoss(reduction='mean')
    R2 = R2Score()
    with torch.no_grad():
        data, target = data[:,0:p].to(DEVICE), data[:,-1].to(DEVICE)


        output1 = net(data)
        out_mpm = net.mpm(data)
   

        target = target.unsqueeze(1).float()
      
    
      
        pears = np.sqrt(R2(output1, target).detach().numpy())
        pears_mpm = np.sqrt(R2(out_mpm, target).detach().numpy())
    
        
      
        RMSE = torch.sqrt(crit(output1,target))
        rmse2 = torch.sqrt(crit(out_mpm,target))
        pinball = mean_pinball_loss(target,output1)
        pinball_mpm = mean_pinball_loss(target,out_mpm)
        
    
        
    

        


        

 
    return RMSE.item(),rmse2.item(),pears.item(),pears_mpm.item(), pinball,pinball_mpm





k = 10

rmse = np.zeros(k)
rmse_mpm = np.zeros(k)
pears = np.zeros(k)
pears_mpm = np.zeros(k)
alphas = np.zeros((k,p))
used_weights = np.zeros(k)
path_length = np.zeros(k)
max_length = np.zeros(k)
pin = np.zeros(k)
pin_mpm = np.zeros(k)

for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = NeuralNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.01)
    scheduler = MultiStepLR(optimizer, milestones=[10000], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        loss = train(net, optimizer)
        scheduler.step()
        
    rmse[i],rmse_mpm[i],pears[i],pears_mpm[i],pin[i],pin_mpm[i] = test_ensemble(net,dtest)
    


    
    layer_names = create_layer_name_list(net = net)
    alp = clean_alpha(net, thresh)
    include_inputs = np.array(include_input_from_layer(alp)) * 1
    alphas[i] = include_inputs.max(axis = 0) ## check if variables have been included from any layer
    density, w, tot_weights = network_density_reduction(alp)
    used_weights[i] = w.item()
    path_length[i] = average_path_length(alp)[0]
    max_length[i] = average_path_length(alp)[1].max()

    

np.savetxt('./abalone-results/alphas_abalone_ann'  +'.txt',alphas, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/rmse_ens_abalone_ann' +'.txt',rmse, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/rmse_med_abalone_ann' +'.txt',rmse_mpm, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/pearson_ens_abalone_ann' +'.txt',pears, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/pearson_med_abalone_ann' +'.txt',pears_mpm, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/pearson_med_abalone_ann_weights' +'.txt',used_weights, delimiter=',',fmt='%s')

np.savetxt('./abalone-results/pinball_ann' +'.txt',pin, delimiter=',',fmt='%s')
np.savetxt('./abalone-results/pinballmpm_ann' +'.txt',pin_mpm, delimiter=',',fmt='%s')





