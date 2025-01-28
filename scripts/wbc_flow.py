#!/usr/bin/env python3
# -*- coding: utf-8 -*-




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
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from utils import ece_score_binary

np.random.seed(1)



X_original, y_original = load_breast_cancer(return_X_y=True)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'
TEST_SAMPLES = 10
TEST_BATCH_SIZE = 57

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 512
epochs = 20000


# Split keep some of the data for validation after training
X, X_test, y, y_test = train_test_split(X_original, y_original, test_size=0.10, random_state=42, stratify=y_original)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

dtest = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
dtrain = torch.tensor(np.column_stack((X,y)),dtype = torch.float32)


TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE


n, p = dtrain.shape

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, num_transforms):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight mu and rho initialization 
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # weight prior is N(0,1) for all the weights
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) + 0.0
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # posterior inclusion initialization
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(0,1))
        
        # inclusion prior is Bernoulli(0.1)
        self.alpha_prior = (self.mu_prior + 0.25).to(DEVICE)
    
        # bias mu and rho initialization
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
    
        # bias prior is also N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.0).to(DEVICE)

        
        # initialization of the flow parameters
        # read MNF paper for more about what this means
        # https://arxiv.org/abs/1703.01961
        self.q0_mean = nn.Parameter(0.001 * torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 +0.001 * torch.randn(in_features))
        self.r0_c = nn.Parameter(0.001 * torch.randn(in_features))
        self.r0_b1 = nn.Parameter(0.001 * torch.randn(in_features))
        self.r0_b2 = nn.Parameter(0.001 * torch.randn(in_features))
        
        #one flow for z and one for r(z|w,gamma)
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)

        self.kl = 0
        self.z = 0

    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z
        zs, log_det_q = self.z_flow(self.z)
        return zs, log_det_q.squeeze()

        # forward path

    def forward(self, input,ensemble = False):
        
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        z_k, _ = self.sample_z()
        ### perform the forward pass 
        if self.training or ensemble:
            
            
        
            e_w = self.weight_mu * self.alpha_q * z_k
            var_w = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z_k) ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps
            
        else:
            w = torch.normal(self.weight_mu * z_k, self.weight_sigma)
            b = torch.normal(self.bias_mu, self.bias_sigma)
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = torch.matmul(input, weight.T) + b


        if self.training:
        ### compute the ELBO
            z2, log_det_q = self.sample_z()
            W_mean = z2 * self.weight_mu * self.alpha_q
            W_var = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z2) ** 2)
            log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                          - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
            log_q = -log_det_q + log_q0
    
            act_mu = self.r0_c @ W_mean.T
            act_var = self.r0_c ** 2 @ W_var.T
            act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
            a = nn.Hardtanh()
            act = a(act_inner)
            mean_r = self.r0_b1.outer(act).mean(-1)  # eq (9) from MNF paper
            log_var_r = self.r0_b2.outer(act).mean(-1)  # eq (10) from MNF paper
            z_b, log_det_r = self.r_flow(z2)
            log_rb = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * log_var_r
                          - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())).sum()
            log_r = log_det_r + log_rb
    
    
            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 + (self.bias_sigma ** 2
                                    + ( self.bias_mu - self.bias_mu_prior) ** 2) / (
                                       2 * self.bias_sigma_prior ** 2)).sum()
    
            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)
                                             - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                                             + (self.weight_sigma ** 2 + (self.weight_mu * z2 - self.mu_prior) ** 2) / (
                                                         2 * self.sigma_prior ** 2))
                             + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()
    
            self.kl = kl_bias + kl_weight + log_q - log_r

        return activations

    # deine the whole BNN


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p - 1, 1, num_transforms=2) #one neuron = logistic regression
        self.loss = nn.BCELoss(reduction='sum')

    def forward(self, x,ensemble=False):
        x = self.l1(x,ensemble)
        x = torch.sigmoid(x) 
        return x

    def kl(self):
        return self.l1.kl


def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:p - 1]
        _y = dtrain[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data,ensemble=True)
        target = target.unsqueeze(1).float()
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
        pred = outputs.squeeze().detach().cpu().numpy()
        pred = np.round(pred, 0)
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)

    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    print('accuracy =', np.mean(accs))
    return negative_log_likelihood.item(), loss.item()


def test_ensemble(net,data):
    net.eval()
    test_nll = nn.BCELoss(reduction='mean')
    with torch.no_grad():
        data, target = data[:,0:(p-1)].to(DEVICE), data[:,-1].to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE,1).to(DEVICE)
        out2 = torch.zeros_like(outputs)
        for i in range(TEST_SAMPLES):

           
            outputs[i] = net(data, ensemble=True)  # model avg over structures and weights
            out2[i] = net(data, ensemble=False)  # only model avg over weights where a > 0.5
        

        output1 = outputs.mean(0)
        out2 = out2.mean(0)
    

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
predicted_alphas = np.zeros(shape=(k, 30)) #store the PiPs here

true_weights = np.array([1,1,0,0])

true_weights = np.array([true_weights, ] * k)

alphas = np.zeros((k,30))
ens =np.zeros(k)
med= np.zeros(k)
ece = np.zeros(k)
ece_mpm = np.zeros(k)
lik = np.zeros(k)
lik_mpm = np.zeros(k)

for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.005)
    scheduler = MultiStepLR(optimizer, milestones=[10000], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        nll, loss = train(net, optimizer)
        scheduler.step()
        
    ens[i],med[i],ece[i],ece_mpm[i],lik[i],lik_mpm[i] = test_ensemble(net,dtest)

    predicted_alphas[i] = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()
    a = net.l1.alpha_q.data.detach().cpu().numpy().squeeze() 
    aa = np.round(a,0)
    tw = (true_weights != 0) * 1
    alphas[i] = net.l1.alpha_q.data
    print(alphas[i])
    





np.savetxt('./wbc-results/alphas_flow'  +'.txt',alphas, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/ens_flow' +'.txt',ens, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/med_flow' +'.txt',med, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/ece_flow'  +'.txt',ece, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/ece_mpm_flow' +'.txt',ece_mpm, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/lik_flow' +'.txt',lik, delimiter=',',fmt='%s')
np.savetxt('./wbc-results/lik_mpm_flow' +'.txt',lik, delimiter=',',fmt='%s')
    
    