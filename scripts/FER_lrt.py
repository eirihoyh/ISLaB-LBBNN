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
from utils import ece_score




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

quit()

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
        self.sigma_prior = (self.mu_prior +  30).to(DEVICE)

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
        self.l1 = BayesianLinear(p-1, 3) #one neuron = logistic regression
     
       

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
        outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE,3).to(DEVICE)
        out2 = torch.zeros_like(outputs)
      
        for i in range(TEST_SAMPLES):

           
            outputs[i] = net(data, sample=True)  # model avg over structures and weights
            out2[i] = net(data, sample=False)  # only model avg over weights where a > 0.5
        
        
        
        

        output1 = outputs.mean(0)
        out2 = out2.mean(0)
        pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
        pred2 = out2.max(1, keepdim=True)[1]

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



alphas = np.zeros((k,6912))
ens_ = np.zeros(k)
median_= np.zeros(k)
eces = np.zeros(k)
eces_mpm = np.zeros(k)
lik =np.zeros(k)
lik_mpm = np.zeros(k)


for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr = 0.005)
    scheduler = MultiStepLR(optimizer, milestones=[1000,2000], gamma=0.1)
    for epoch in range(epochs):
        
        print('epoch =', epoch)
        nll, loss = train(net, optimizer)
        scheduler.step()
        
    ens_[i],median_[i],eces[i],eces_mpm[i],lik[i],lik_mpm[i] = test_ensemble(net,dtest)
  
    alphas[i] = net.l1.alpha_q.data.flatten()
    
    
np.savetxt('./fer-results/alphas_lrt'  +'.txt',alphas, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ens_lrt' +'.txt',ens_, delimiter=',',fmt='%s')
np.savetxt('./fer-results/med_lrt' +'.txt',median_, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ece_lrt'  +'.txt',eces, delimiter=',',fmt='%s')
np.savetxt('./fer-results/ecempm_lrt' +'.txt',eces_mpm, delimiter=',',fmt='%s')
np.savetxt('./fer-results/lik_lrt' +'.txt',lik, delimiter=',',fmt='%s')
np.savetxt('./fer-results/likmpm_lrt' +'.txt',lik_mpm, delimiter=',',fmt='%s')
quit()
    
def plot_model_vision_image(net, train_data, train_target, c=0, net_nr=0, threshold=0.5, thresh_w=0.0, save_path=None):
    '''
    NOTE: Works just for quadratic images atm, should probably generalize to prefered
            dim at a later point
    '''
    
    colors = ["white", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    
    clean_a = net.l1.alpha_q

    p = 48
    img_avg = np.zeros(p*p)

    w = net.l1.weight_mu[-1][c, -p*p:].detach().numpy()
    w = np.where(clean_a[-1][c,-p*p:].detach().numpy() == 1, w, 0)
    
    avg_c_img = train_data[train_target == c].mean(axis=0).reshape((p,p))

    fig, axs = plt.subplots(len(clean_a)+1, figsize=(10,10))
    
    for ind, ca in enumerate(clean_a):
        out = ca.shape[0]
        img_layer = np.zeros(p*p)
        for j in range(out):
            # img_layer += ca[j,-p:].detach().numpy()
            img_layer += np.where(np.abs(w) >= thresh_w, ca[j,-p*p:].detach().numpy(), 0)

        img_avg += img_layer
        axs[ind].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
        if np.sum(img_layer) > 0:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5)#, vmin=min_max*-1, vmax=min_max*1)
        else:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            
        fig.colorbar(im, ax=axs[ind])
        axs[ind].set_title(f"Class {c}, Layer {ind}")
        axs[ind].set_xticks([])
        axs[ind].set_yticks([])
        

    # min_max = max(np.concatenate((img_pos, img_neg*-1)))
    min_max = max(np.concatenate((img_avg, img_avg*-1)))

    
    axs[ind+1].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
    im = axs[ind+1].imshow(img_avg.reshape((p,p)), cmap=cmap, alpha=0.5, vmin=0, vmax=min_max*1)
    axs[ind+1].set_title(f"Net: {net_nr} all layers")
    axs[ind+1].set_xticks([])
    axs[ind+1].set_yticks([])
    fig.colorbar(im, ax=axs[ind+1])
    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
    
    
cl = X[y == 0]
cl = cl.reshape(len(cl), 48,48)
clmean = cl.mean(axis = 0)
plt.imshow(clmean,cmap = 'Greys',alpha = 0.5)

alpha = (net.l1.alpha_q[0] > 0.5) * 1
alpha = alpha.reshape(48,48)
xx,yy = np.where(alpha  == 1)

plt.scatter(x=xx, y=yy, c='r', s=5)
plt.show()

cl = X[y == 1]
cl = cl.reshape(len(cl), 48,48)
clmean = cl.mean(axis = 0)
plt.imshow(clmean,cmap = 'Greys',alpha = 0.5)

alpha = (net.l1.alpha_q[1] > 0.5) * 1
alpha = alpha.reshape(48,48)
xx,yy = np.where(alpha  == 1)

plt.scatter(x=xx, y=yy, c='r', s=5)
plt.show()


cl = X[y == 2]
cl = cl.reshape(len(cl), 48,48)
clmean = cl.mean(axis = 0)
plt.imshow(clmean,cmap = 'Greys',alpha = 0.5)

alpha = (net.l1.alpha_q[2] > 0.5) * 1
alpha = alpha.reshape(48,48)
xx,yy = np.where(alpha  == 1)

plt.scatter(x=xx, y=yy, c='r', s=5)
plt.show()






