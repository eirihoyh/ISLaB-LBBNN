import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.colors as mcolors
from utils import (nr_hidden_layers,get_alphas,clean_alpha, 
include_input_from_layer, network_density_reduction,create_layer_name_list,
average_path_length,ece_score)




from torch.optim.lr_scheduler import MultiStepLR

import xlrd
np.random.seed(1)


df = pd.read_csv("icml_face_data.csv")

data = np.zeros((len(df), 48*48))
for i in range(len(data)):
    data[i] = np.array(df[" pixels"][i].split(" ")).astype(np.float32)/255.
targets = df.emotion.values

X,y  = data, targets
used_inds = [yi in [3,5,6] for yi in y]
y = y[used_inds]
y = np.where(y == 3, 0, y)
y = np.where(y == 5, 1, y)
y = np.where(y == 6, 2, y)
X = X[used_inds]


X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y)








dtest = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
dtrain = torch.tensor(np.column_stack((X,y)),dtype = torch.float32)

  

BATCH_SIZE =1727
epochs = 1000
TEST_SAMPLES = 10
TEST_BATCH_SIZE = 1919

TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE
TEST_SIZE = len(dtest)


n, p = X.shape

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


dim = 200
HIDDEN_LAYERS = 2
thresh = 0.005






class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(p, dim)])
        self.linears.extend([nn.Linear((dim + p), (dim)) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(nn.Linear((dim+p), 3))


    def forward(self, x):
       x_input = x.view(-1, p)
       x = torch.sigmoid(self.linears[0](x_input))
       i = 1
       for l in self.linears[1:-1]:
           x = torch.sigmoid(l(torch.cat((x, x_input),1)))
           i += 1

       out = self.linears[i](torch.cat((x, x_input),1))
       return F.log_softmax(out,dim = 1)
   
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
  
       
       return F.log_softmax(preact,dim = 1)
   
    
   
    
def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        
        _x = dtrain[old_batch: batch_size * batch, 0:p]
        _y = dtrain[old_batch: batch_size * batch, -1]
 
        old_batch = batch_size * batch
        target = _y.type(torch.LongTensor).to(DEVICE)
        data = _x.to(DEVICE)
        net.zero_grad()
        outputs = net(data)
        nll = F.nll_loss(outputs, target, reduction="sum")
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
    

    with torch.no_grad():
        data, target = data[:,0:(p)].to(DEVICE), data[:,-1].to(DEVICE)
        output1 = net(data)
        output2 = net.mpm(data)  # mpm
        
        
        
        

    
        pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
        pred2 = output2.max(1, keepdim=True)[1]

        median = pred2.eq(target.view_as(pred2)).sum().item() / TEST_SIZE
        ensemble = pred1.eq(target.view_as(pred1)).sum().item() / TEST_SIZE
        
        full_nll = F.nll_loss(output1, target.long(), reduction="mean")
        sparse_nll = F.nll_loss(output2, target.long(), reduction="mean")
        
        o= output1.detach().cpu().numpy()
        o2= output2.detach().cpu().numpy()
        tar = target.detach().cpu().numpy()

        ece = (ece_score(np.exp(o),tar))
         
        ece_mpm = (ece_score(np.exp(o2),tar))

    
       




        

 
    return ensemble,median,ece,ece_mpm,full_nll.detach().cpu().numpy(),sparse_nll.detach().cpu().numpy()



k = 10

alphas = np.zeros((k,616))
ens_ = np.zeros(k)
median_= np.zeros(k)
used_weights = np.zeros(k)
path_length = np.zeros(k)
max_length = np.zeros(k)
eces = np.zeros(k)
eces_mpm = np.zeros(k)
lik = np.zeros(k)
lik_mpm = np.zeros(k)

for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = NeuralNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.005)
    scheduler = MultiStepLR(optimizer, milestones=[1000,2000], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        loss = train(net, optimizer)
        scheduler.step()
        
    ens_[i],median_[i],eces[i],eces_mpm[i],lik[i],lik_mpm[i] = test_ensemble(net,dtest)
    


    
    layer_names = create_layer_name_list(net = net)
    alp = clean_alpha(net, thresh)
    include_inputs = np.array(include_input_from_layer(alp)) * 1
   # alphas[i] = include_inputs.max(axis = 0) ## check if variables have been included from any layer
    density, w, tot_weights = network_density_reduction(alp)
    used_weights[i] = w.item()
    path_length[i] = average_path_length(alp)[0]
    max_length[i] = average_path_length(alp)[1].max()

    
np.savetxt('./fer-results/used_weights_ann'  +'.txt',used_weights, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ens_ann' +'.txt',ens_, delimiter=',',fmt='%s')
np.savetxt('./fer-results/med_ann' +'.txt',median_, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ece_ann'  +'.txt',eces, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ecempm_ann' +'.txt',eces_mpm, delimiter=',',fmt='%s')
np.savetxt('./fer-results/lik_ann' +'.txt',lik, delimiter=',',fmt='%s')
np.savetxt('./fer-results/likmpm_ann' +'.txt',lik_mpm, delimiter=',',fmt='%s')









