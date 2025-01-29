




# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pandas as pd
from flows import PropagateFlow
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
import xlrd
from utils import ece_score
np.random.seed(1)

df = pd.read_excel("Data_Cortex_Nuclear.xls")

X_pd = df.iloc[:,1:-4]
X_pd.fillna(X_pd.mean(),inplace=True)
X = X_pd.values
target = df.values[:, -1]


n, p = X.shape  # need this to get p 
y = np.zeros(len(target))
labeling = {}
for i, val in enumerate(np.unique(target)):
    labeling[i] = val
    y[target==val] = i



X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y)

mu,std = X.mean(axis = 0), X.std(axis = 0)
X = (X - mu) / std
X_test = (X_test - mu) / std


dtest = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
dtrain = torch.tensor(np.column_stack((X,y)),dtype = torch.float32)


#savedir = '/Users/larsskaaret-lund/Documents/Eirik paper code/data/mice'
#np.savetxt(savedir + '/X_train.txt',X, delimiter=',')
#np.savetxt(savedir + '/X_test.txt',X_test, delimiter=',')
#np.savetxt(savedir + '/Y_train.txt',y, delimiter=',')
#np.savetxt(savedir + '/Y_test.txt',y_test, delimiter=',')



BATCH_SIZE =972
epochs = 20000
TEST_SAMPLES = 10
TEST_BATCH_SIZE = 108

TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE
TEST_SIZE = len(dtest)


n, p = dtrain.shape





DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight_sigma = torch.empty(size = self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) 
        self.sigma_prior = (self.mu_prior +  25).to(DEVICE)

        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(0,1))
        self.alpha_q = torch.empty(size = self.lambdal.shape)

        # prior inclusion probability
        self.alpha_prior = (self.mu_prior + 0.05).to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or sample:
            e_w = self.weight_mu * self.alpha_q
            var_w = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:  
            w = torch.normal(self.weight_mu , self.weight_sigma)
            b = torch.normal(self.bias_mu, self.bias_sigma)
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = torch.matmul(input, weight.T) + b

        if self.training or calculate_log_probs:



            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                    + (self.bias_mu - self.bias_mu_prior) ** 2) / (
                               2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)
                                         - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                                         + (self.weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2) / (
                                                     2 * self.sigma_prior ** 2))
                         + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0

        return activations






class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p-1, 8) #one neuron = logistic regression
     
       

    def forward(self, x,sample=False):
        x = self.l1(x,sample)
       
        return F.log_softmax(x,dim = 1)

    def kl(self):
        return self.l1.kl


def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
   
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:(p-1)]
        _y = dtrain[old_batch: batch_size * batch, -1]
     

        old_batch = batch_size * batch
        target = _y.type(torch.LongTensor).to(DEVICE)
     
        data = _x.to(DEVICE)
        net.zero_grad()
        outputs = net(data,sample=True)
     
    
        

        print(outputs)
        print(target)
        quit()
    
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="sum")
        loss = negative_log_likelihood + (net.kl() / NUM_BATCHES)  
        loss.backward()
        optimizer.step()


    print('loss', loss.item(),'nll',negative_log_likelihood.item())
 

    return negative_log_likelihood.item(), loss.item()


def test_ensemble(net,data):
    net.eval()
    

    with torch.no_grad():
        data, target = data[:,0:(p-1)].to(DEVICE), data[:,-1].to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE,8).to(DEVICE)
        out2 = torch.zeros_like(outputs)
      
        for i in range(TEST_SAMPLES):

           
            outputs[i] = net(data, sample=True)  # model avg over structures and weights
            out2[i] = net(data, sample=False)  # only model avg over weights where a > 0.5
        
        
        
        

        output1 = outputs.mean(0)
        out2 = out2.mean(0)
        pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
        pred2 = out2.max(1, keepdim=True)[1]
        print(pred1)
        quit()
        median = pred2.eq(target.view_as(pred2)).sum().item() / TEST_SIZE
        ensemble = pred1.eq(target.view_as(pred1)).sum().item() / TEST_SIZE
        
        
        full_nll = F.nll_loss(output1, target.long(), reduction="mean")
        sparse_nll = F.nll_loss(out2, target.long(), reduction="mean")
        
        o= output1.detach().cpu().numpy()
        o2= out2.detach().cpu().numpy()
        tar = target.detach().cpu().numpy()

        ece = (ece_score(np.exp(o),tar))
         
        ece_mpm = (ece_score(np.exp(o2),tar))
     
    
       




        

 
    return ensemble,median,ece,ece_mpm,full_nll.detach().cpu().numpy(),sparse_nll.detach().cpu().numpy()



k = 10
predicted_alphas = np.zeros(shape=(k, p-1)) #store the PiPs here



alphas = np.zeros((k,616))
ens_ = np.zeros(k)
median_= np.zeros(k)
eces = np.zeros(k)
eces_mpm = np.zeros(k)
lik = np.zeros(k)
lik_mpm = np.zeros(k)


for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.05)
    scheduler = MultiStepLR(optimizer, milestones=[5000,10000,15000], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        nll, loss = train(net, optimizer)
        scheduler.step()
        
    ens_[i],median_[i],eces[i],eces_mpm[i],lik[i],lik_mpm[i] = test_ensemble(net,dtest)
   
    alphas[i] = net.l1.alpha_q.data.flatten()

    




np.savetxt('./mice-results/alphas_lrt'  +'.txt',alphas, delimiter=',',fmt='%s')
np.savetxt('./mice-results/ens_lrt' +'.txt',ens_, delimiter=',',fmt='%s')
np.savetxt('./mice-results/med_lrt' +'.txt',median_, delimiter=',',fmt='%s')
np.savetxt('./mice-results/ece_lrt' +'.txt',eces, delimiter=',',fmt='%s')
np.savetxt('./mice-results/ecempm_lrt' +'.txt',eces_mpm, delimiter=',',fmt='%s')
np.savetxt('./mice-results/lik_lrt' +'.txt',lik, delimiter=',',fmt='%s')
np.savetxt('./mice-results/likmpm_lrt' +'.txt',lik_mpm, delimiter=',',fmt='%s')
