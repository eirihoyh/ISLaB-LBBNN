import copy
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from config import config
import os
import sys
current_dir = os.getcwd()

sys.path.append('islbbnn')
import plot_functions as pf
import pipeline_functions as pip_func
sys.path.append('islbbnn/networks')
from lrt_net import BayesianNetwork
import torch.nn.functional as F

os.chdir(current_dir) # set the working directory back to this one 


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

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




X_train_original = np.loadtxt("data/abalone/X_train.txt", delimiter=",")
X_test_original = np.loadtxt("data/abalone/X_test.txt", delimiter=",")
y_train_original = np.loadtxt("data/abalone/Y_train.txt", delimiter=",")
y_test_original = np.loadtxt("data/abalone/Y_test.txt", delimiter=",")

n,p = X_train_original.shape
print(n,p,dim)

# assert (TRAIN_SIZE % BATCH_SIZE) == 0


test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)

# Load trained network
net = torch.load(f"implementations/local_explain/abalone/network/net0", weights_only=False,map_location=torch.device('cpu'))


device = torch.device("cpu")
net.to(device)

# Abalone variable names
variable_names = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Sex_I", "Sex_M"]

# i = 101
# i = 200
# i = 106
i = 201
explain_this = test_dat[i,:-1].reshape(-1, p)[0]
output_val = test_dat[i,-1]
print(f"True output value: {int(output_val)}")
pf.plot_local_explain_piecewise_linear_act(
    net, 
    explain_this, 
    variable_names=copy.deepcopy(variable_names), 
    n_samples=1_000,
    magnitude=True, 
    include_prediction=True, 
    include_bias=True, 
    fig_size=(6.4, 4.8), 
    no_zero_contributions=False,
    include_potential_contribution=False,
    ann=False, 
    save_path=f"implementations/local_explain/abalone/imgs/local_explain_{int(output_val)}")

# # Empirical explanation
# explain_this = explain_this.reshape(-1, p)
# pf.plot_local_contribution_empirical(
#     net, 
#     explain_this, 
#     sample=True, 
#     median=True, 
#     n_samples=1_000,
#     variable_names=copy.deepcopy(variable_names), 
#     class_names=["regression model"], 
#     magnitude=True,
#     save_path=f"implementations/local_explain/abalone/imgs/local_explain_{int(output_val)}_emp_approach")
