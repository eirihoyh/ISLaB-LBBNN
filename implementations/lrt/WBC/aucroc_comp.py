import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import config
import os
import sys
current_dir = os.getcwd()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append('islbbnn')
import plot_functions as pf
import pipeline_functions as pip_func
sys.path.append('islbbnn/networks')
from lrt_net import BayesianNetwork
import torch.nn.functional as F

os.chdir(current_dir) # set the working directory back to this one 


# define parameters
HIDDEN_LAYERS = config['n_layers'] - 2 
epochs = config['num_epochs']
post_train_epochs = config['post_train_epochs']
dim = config['hidden_dim']
num_transforms = config['num_transforms']
n_nets = config['n_nets']
n_samples = config['n_samples']
lr = config['lr']
class_problem = config["class_problem"]
non_lin = config["non_lin"]
verbose = config['verbose']
save_res = config['save_res']
patience = config['patience']
alpha_prior = config['inclusion_prob_prior']
std_prior = config['std_prior']
lower_init_lambda = config['lower_init_lambda']
upper_init_lambda = config['upper_init_lambda']
high_init_covariate_prob = config['high_init_covariate_prob']

# Get data
X_train_original = np.loadtxt(f"data/WBC/X_train.txt", delimiter=",")
X_test_original = np.loadtxt(f"data/WBC/X_test.txt", delimiter=",")
y_train_original = np.loadtxt(f"data/WBC/Y_train.txt", delimiter=",")
y_test_original = np.loadtxt(f"data/WBC/Y_test.txt", delimiter=",")

n, p = X_train_original.shape  # need this to get p 
print(n,p,dim)

if class_problem:
    n_classes = len(np.unique(y_train_original))
    if n_classes == 2:
        n_classes = 1
    multiclass = n_classes > 1
else: 
    n_classes = 1  # Just need to set it to something above zero
    multiclass = False

BATCH_SIZE = int((n)/8)
TRAIN_SIZE = int((n))


NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

print(NUM_BATCHES)

train_dat = torch.tensor(np.column_stack((X_train_original,y_train_original)),dtype = torch.float32)
test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)



def evaluate(model, test_data, samples=100):
    model.eval()

    _x = test_data[:, :-1]
    _y = test_data[:, -1]

    with torch.no_grad():
        images, labels = _x.to(DEVICE),_y.type(torch.LongTensor).to(DEVICE)
        
        outputs_median = torch.zeros(samples, images.shape[0], n_classes).to(DEVICE)
        outputs_full = torch.zeros(samples, images.shape[0], n_classes).to(DEVICE)
        for i in range(samples):
            outputs_median[i] = model.forward(images, ensemble=False, sample=True, calculate_log_probs=True)
            outputs_full[i] = model.forward(images, ensemble=True, sample=True, calculate_log_probs=True)

        outputs_median = outputs_median.mean(0)
        roc_auc_median = roc_auc_score(labels.detach().cpu().numpy(), outputs_median.detach().cpu().numpy())
        outputs_full = outputs_full.mean(0)
        roc_auc_full = roc_auc_score(labels.detach().cpu().numpy(), outputs_full.detach().cpu().numpy())
        

    return roc_auc_median, roc_auc_full



results = []

for i in range(n_nets):  # Use the list from wbc_lrt.py
    print(f"Testing net {i}")
    model = torch.load(f"implementations/lrt/WBC/network/net{i}", weights_only=False)#,map_location=torch.device('cpu'))
    model.to(DEVICE)


    roc_auc_median, roc_auc_full = evaluate(model, test_dat)



    print(f"roc_auc median: {roc_auc_median:.4f}, roc_auc full: {roc_auc_full:.4f}")
    nr_active_weights = 0
    nr_weights_total = 0
    for name, param in model.linears.named_parameters():
        if f"lambdal" in name:
            active_weights = copy.deepcopy(1 / (1 + np.exp(-param.cpu().data))).cpu().detach().numpy() > 0.5
            nr_active_weights += np.sum(active_weights)
            nr_weights_total += np.sum(np.prod(active_weights.shape))

    print(f"Used weights: {nr_active_weights}, Total weights: {nr_weights_total}, Total density: {(nr_active_weights/nr_weights_total):.4f}")
    results.append({
        'network': i,
        'roc_auc_median': roc_auc_median.item(),
        'roc_auc_full': roc_auc_full.item(),
        'nr_active_weights': nr_active_weights
    })

df = pd.DataFrame(results)
df.to_csv('implementations/lrt/WBC/results/lrt_networks_roc_auc_results.csv', index=False)
