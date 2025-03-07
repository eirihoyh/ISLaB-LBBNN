import torchvision
from torchvision import transforms
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

# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# cuda = torch.cuda.set_device(0)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")


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

#---------DATA------------
trainset = torchvision.datasets.MNIST(
    root='./mnist', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(
    root='./mnist', train=False, download=True, transform=transforms.ToTensor())

train_data = trainset.data/255.
train_target = trainset.targets

test_data = (testset.data/255.)
test_target = testset.targets

print(test_target)

p = train_data.shape[-1]

X_train_original = train_data.detach().numpy().reshape(-1,p*p)
X_test_original = test_data.detach().numpy().reshape(-1,p*p)
y_train_original = train_target.detach().numpy()
y_test_original = test_target.detach().numpy()

n,p = X_train_original.shape
print(n,p,dim)

n_classes = len(np.unique(y_train_original))


test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)

# Load trained network
net = torch.load(f"implementations/local_explain/mnist/network/net0_dense_init", weights_only=False,map_location=torch.device('cpu'))


## Get contribution plots
# train_data = torch.tensor(copy.deepcopy(X_train_original),dtype=torch.float32)
# train_target = torch.tensor(copy.deepcopy(y_train_original), dtype=torch.float32)
# for c in range(n_classes):
    # pf.plot_model_vision_image(net, train_data=train_data, train_target=train_target, c=c, save_path=f"implementations/local_explain/mnist/contribution_plots/class{c}")


# Explain 4 digit
explain_this = train_data[26].reshape(-1,p)
# Grad approach
pf.plot_local_contribution_images_contribution_gradient_approach(net, explain_this[0], n_classes=n_classes, n_samples=100, save_path="implementations/local_explain/mnist/local_plots/digit4")
print("Fin grad approach digit 4")
# Empirical explain
pf.plot_local_contribution_images_contribution_empirical_magnitude(net, explain_this, n_classes=n_classes, n_samples=100, save_path="implementations/local_explain/mnist/local_plots/digit4")
print("Fin emp approach digit 4")

# Explain 7 digit
explain_this = train_data[96].reshape(-1,p)

# Grad approach
pf.plot_local_contribution_images_contribution_gradient_approach(net, explain_this[0], n_classes=n_classes, n_samples=100, save_path="implementations/local_explain/mnist/local_plots/digit7")
print("Fin grad approach digit 7")
# Empirical explain
pf.plot_local_contribution_images_contribution_empirical_magnitude(net, explain_this, n_classes=n_classes, n_samples=100, save_path="implementations/local_explain/mnist/local_plots/digit7")
print("Fin emp approach digit 7")