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
from flow_cnn_net import BCNN
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
lr = config['lr']
class_problem = config["class_problem"]
verbose = config['verbose']
save_res = config['save_res']
patience = config['patience']
alpha_prior = config['inclusion_prob_prior']
std_prior = config['std_prior']
lower_init_lambda = config['lower_init_lambda']
upper_init_lambda = config['upper_init_lambda']
high_init_covariate_prob = config['high_init_covariate_prob']
out_channels_list = config['out_channels_list']
kernel_size = config['kernel_size']
stride = config['stride']
padding = config['padding']

#---------DATA------------
trainset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=True, transform=transforms.ToTensor())

train_data = trainset.data/255.
train_target = trainset.targets

test_data = (testset.data/255.)
test_target = testset.targets

print(np.unique(test_target))

print(train_data.shape)
print(type(train_data))

n, p1, p2, channels = train_data.shape

p = channels*p1*p2

X_train_original = train_data.reshape(-1,p)
X_test_original = test_data.reshape(-1,p)
y_train_original = train_target
y_test_original = test_target

# n,p1,p2 = X_train_original.shape
print(n,channels,p1,p2,dim)

n_classes = len(np.unique(y_train_original))
multiclass = n_classes > 1

# BATCH_SIZE = int((n)/10)
BATCH_SIZE = int((n)/500)
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
    net = BCNN(
        channels, 
        out_channels_list, 
        kernel_size, 
        stride, 
        padding, 
        p1, 
        p2, 
        dim, 
        HIDDEN_LAYERS, 
        a_prior=alpha_prior, 
        std_prior=std_prior, 
        num_transforms=num_transforms,
        n_classes=n_classes,
        act_func=F.leaky_relu,
        lower_init_lambda=lower_init_lambda,
        upper_init_lambda=upper_init_lambda,
        high_init_covariate_prob=high_init_covariate_prob).to(DEVICE)
    alphas = pip_func.get_alphas_numpy(net)
    nr_weights = np.sum([np.prod(a.shape) for a in alphas])
    print(nr_weights)

    params = []
    for name, param in net.named_parameters():
        if f"lambdal" in name:
            alpha_lr = {'params': param, 'lr': .01}
            params.append(alpha_lr)
        else:
            param_lr = {'params': param, 'lr': lr}
            params.append(param_lr)

    # print(params)
    
    optimizer = optim.Adam(params, lr=lr)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    
    scheduler = MultiStepLR(optimizer, milestones=[int(0.7*tot_rounds), int(0.9*tot_rounds)], gamma=0.7) # int(0.3*tot_rounds), int(0.5*tot_rounds),

    all_nll = []
    all_loss = []

    # Split into training and test set
    # X_train, X_val, y_train, y_val = train_test_split(
    # X, y, test_size=1/9, random_state=ni, stratify=y)
            
    train_dat = torch.tensor(np.column_stack((X_train_original,y_train_original)),dtype = torch.float32)
    # val_dat = torch.tensor(np.column_stack((X_val,y_val)),dtype = torch.float32)

    for name, param in net.named_parameters():
        if f"lambdal" in name:
            param.requires_grad_(False)

    
    # Train network
    counter = 0
    highest_acc = 0
    best_model = copy.deepcopy(net)
    for epoch in range(tot_rounds):
        if verbose:
            print(epoch)
        nll, loss = pip_func.train(net, train_dat, optimizer, BATCH_SIZE, NUM_BATCHES, p, DEVICE, nr_weights, post_train=post_train, multiclass=multiclass)
        all_nll.append(nll)
        all_loss.append(loss)
        scheduler.step()
        
        nll_val, loss_val, ensemble_val = pip_func.val(net, test_dat, DEVICE, verbose=verbose, reg=(not class_problem), multiclass=multiclass)
        if ensemble_val >= highest_acc:
            counter = 0
            highest_acc = ensemble_val
            best_model = copy.deepcopy(net)
        else:
            counter += 1
        
        

        if epoch == epochs-1:
            post_train = True   # Post-train --> use median model 
            for name, param in net.named_parameters():
                if f"lambdal" in name:
                    param.requires_grad_(False)

        if counter >= patience:
            break
        if epoch == 2:
            for name, param in net.named_parameters():
                if f"lambdal" in name:
                    param.requires_grad_(True)
        
        nr_active_weights = 0
        nr_weights_total = 0
        for name, param in net.named_parameters():
            if f"lambdal" in name:
                active_weights = copy.deepcopy(1 / (1 + np.exp(-param.cpu().data))).cpu().detach().numpy() > 0.5
                nr_active_weights += np.sum(active_weights)
                nr_weights_total += np.sum(np.prod(active_weights.shape))

        print(f"Used weights: {nr_active_weights}, Total weights: {nr_weights_total}, Total density: {(nr_active_weights/nr_weights_total):.4f}")

    all_nets[ni] = net 
    # Results
    if save_res:
        torch.save(net, f"comparison_scripts/flow_cnn/cifar10/network/net{ni}")
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
    np.savetxt(f'comparison_scripts/flow_cnn/cifar10/results/flow_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_cifar10_leaky_relu_full.txt',m,delimiter = ',')
    # m_median = np.array(metrics_median_several_runs)
    np.savetxt(f'comparison_scripts/flow_cnn/cifar10/results/flow_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_cifar10_leaky_relu_median.txt',m_median,delimiter = ',')