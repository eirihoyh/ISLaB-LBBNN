import torchvision
from torchvision import transforms
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from config_dense_init import config
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
multiclass = n_classes > 1

BATCH_SIZE = int((n)/50)
# TEST_BATCH_SIZE = int(n*0.10) # Would normally call this the "validation" part (will be used during training)
# VAL_BATCH_SIZE = int(n*0.10) # and this the "test" part (will be used after training)

TRAIN_SIZE = int((n))


NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

assert (TRAIN_SIZE % BATCH_SIZE) == 0


# Split keep some of the data for validation after training
# X, X_test, y, y_test = train_test_split(
#     copy.deepcopy(X_original), copy.deepcopy(y_original), test_size=0.10, random_state=42, stratify=y_original)

test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)

tot_rounds = epochs + post_train_epochs

all_nets = {}
metrics_several_runs = []
metrics_median_several_runs = []
for ni in range(n_nets):
    post_train = False
    print('network', ni)
    # Initate network
    torch.manual_seed(ni+42)
    net = BayesianNetwork(dim, p, HIDDEN_LAYERS, classification=class_problem, a_prior=alpha_prior, std_prior=std_prior, n_classes=n_classes).to(DEVICE)
    alphas = pip_func.get_alphas_numpy(net)
    nr_weights = np.sum([np.prod(a.shape) for a in alphas])
    print(nr_weights)

    params = []
    for name, param in net.named_parameters():
        if f"lambdal" in name:
            alpha_lr = {'params': param, 'lr': 0.1}
            params.append(alpha_lr)
        else:
            param_lr = {'params': param, 'lr': lr}
            params.append(param_lr)

    # print(params)
    
    optimizer = optim.Adam(params, lr=lr)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    
    scheduler = MultiStepLR(optimizer, milestones=[int(0.3*tot_rounds), int(0.5*tot_rounds), int(0.7*tot_rounds), int(0.9*tot_rounds)], gamma=0.7)

    all_nll = []
    all_loss = []

    # Split into training and test set
    # X_train, X_val, y_train, y_val = train_test_split(
    # X, y, test_size=1/9, random_state=ni, stratify=y)
            
    train_dat = torch.tensor(np.column_stack((X_train_original,y_train_original)),dtype = torch.float32)
    # val_dat = torch.tensor(np.column_stack((X_val,y_val)),dtype = torch.float32)
    
    # Train network
    counter = 0
    highest_acc = 0
    best_model = copy.deepcopy(net)
    for epoch in range(tot_rounds):
        if verbose:
            print(epoch)
        nll, loss = pip_func.train(net, train_dat, optimizer, BATCH_SIZE, NUM_BATCHES, p, DEVICE, nr_weights, post_train=post_train, multiclass=multiclass)
        nll_val, loss_val, ensemble_val = pip_func.val(net, test_dat, DEVICE, verbose=verbose, reg=(not class_problem), multiclass=multiclass)
        if ensemble_val >= highest_acc:
            counter = 0
            highest_acc = ensemble_val
            best_model = copy.deepcopy(net)
        else:
            counter += 1
        
        all_nll.append(nll)
        all_loss.append(loss)

        if epoch == epochs-1:
            post_train = True   # Post-train --> use median model 
            for name, param in net.named_parameters():
                for i in range(HIDDEN_LAYERS+1):
                    #if f"linears{i}.lambdal" in name:
                    if f"linears.{i}.lambdal" in name:
                        param.requires_grad_(False)

        if counter >= patience:
            break

        scheduler.step()
        
    all_nets[ni] = net 
    # Results
    if save_res:
        torch.save(net, f"implementations/lrt/MNIST/network/net{ni}_dense_init")
    metrics, metrics_median = pip_func.test_ensemble(all_nets[ni], test_dat, DEVICE, SAMPLES=100, CLASSES=n_classes, reg=(not class_problem), multiclass=multiclass) # Test same data 10 times to get average 
    metrics_several_runs.append(metrics)
    metrics_median_several_runs.append(metrics_median)

if verbose:
    print(metrics)
m = np.array(metrics_several_runs)
m_median = np.array(metrics_median_several_runs)

print("Results full model:")
print(m)
print("\n\nResults median prob model:")
print(m_median)

if save_res:
    # m = np.array(metrics_several_runs)
    np.savetxt(f'implementations/lrt/MNIST/results/lrt_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_mnist_sigmoid_dense_init_full.txt',m,delimiter = ',')
    # m_median = np.array(metrics_median_several_runs)
    np.savetxt(f'implementations/lrt/MNIST/results/lrt_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_mnist_sigmoid_dense_init_median.txt',m_median,delimiter = ',')