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


dep = 0.0


# Get linear data
X_train_original = np.loadtxt(f"data/linear/{dep}X_train.txt", delimiter=",")
X_test_original = np.loadtxt(f"data/linear/{dep}X_test.txt", delimiter=",")
y_train_original = np.loadtxt(f"data/linear/{dep}Y_train.txt", delimiter=",")
y_test_original = np.loadtxt(f"data/linear/{dep}Y_test.txt", delimiter=",")

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

BATCH_SIZE = int((n)/50)
TRAIN_SIZE = int((n))


NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

print(NUM_BATCHES)

test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)


tot_rounds = epochs + post_train_epochs

# select the device and initiate model

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
all_nets = {}
metrics_several_runs = []
metrics_median_several_runs = []
for ni in range(n_nets):
    post_train = False
    print('network', ni)
    # Initate network
    torch.manual_seed(ni+42)
    net = BayesianNetwork(
        dim, 
        p, 
        HIDDEN_LAYERS, 
        classification=class_problem, 
        a_prior=alpha_prior, 
        std_prior=std_prior, 
        n_classes=n_classes, 
        act_func=F.relu,
        lower_init_lambda=lower_init_lambda,
        upper_init_lambda=upper_init_lambda,
        high_init_covariate_prob=high_init_covariate_prob).to(DEVICE)
    alphas = pip_func.get_alphas_numpy(net)
    nr_weights = np.sum([np.prod(a.shape) for a in alphas])
    print(nr_weights)

    # params = []
    # for name, param in net.named_parameters():
    #     if f"lambdal" in name:
    #         alpha_lr = {'params': param, 'lr': 1.5}
    #         params.append(alpha_lr)
    #     else:
    #         param_lr = {'params': param, 'lr': lr}
    #         params.append(param_lr)

    # # print(params)
    # optimizer = optim.Adam(params, lr=lr)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    
    scheduler = MultiStepLR(optimizer, milestones=[int(0.7*tot_rounds), int(0.9*tot_rounds)], gamma=0.5)

    all_nll = []
    all_loss = []

    train_dat = torch.tensor(np.column_stack((X_train_original,y_train_original)),dtype = torch.float32)
    
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
    
    if save_res:
        torch.save(net, f"implementations/local_explain/linear_problem/network/net{ni}_dep_level_{int(dep*100)}")
    all_nets[ni] = net 
    # Results
    metrics, metrics_median = pip_func.test_ensemble(all_nets[ni], test_dat, DEVICE, SAMPLES=100, CLASSES=n_classes, reg=(not class_problem)) # Test same data 10 times to get average 
    metrics_several_runs.append(metrics)
    metrics_median_several_runs.append(metrics_median)
    pf.run_path_graph(all_nets[ni], threshold=0.5, save_path=f"implementations/local_explain/linear_problem/path_graphs/prob/net{ni}_dep_level_{int(dep*100)}_relu", show=False)
    pf.run_path_graph_weight(net, save_path=f"implementations/local_explain/linear_problem/path_graphs/weight/net{ni}_dep_level_{int(dep*100)}_relu", show=False)

if verbose:
    print(metrics)
m = np.array(metrics_several_runs)
m_median = np.array(metrics_median_several_runs)

print("Results full model:")
print(m)
print("\n\nResults median prob model:")
print(m_median)


device = torch.device("cpu")
net.to(device)

variable_names=["x1", "x2", "x3", "x4"]

explain_this = torch.tensor([5.50, -5.50, 10.0, -50.0])
pf.plot_local_explain_piecewise_linear_act(
    net, 
    explain_this, 
    n_samples=1_000,
    magnitude=True, 
    include_prediction=True, 
    include_bias=True, 
    fig_size=(6.4, 4.8), 
    no_zero_contributions=False, 
    ann=False, 
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_equal_contribution")

# Empirical explanation
explain_this = explain_this.reshape(-1, p)
pf.plot_local_contribution_empirical(
    net, 
    explain_this, 
    sample=True, 
    median=True, 
    n_samples=1_000,
    variable_names=copy.deepcopy(variable_names), 
    class_names=["binary prediction"], 
    magnitude=True,
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_equal_contribution_emp_approach")


explain_this = torch.tensor([1.50, 1.50, 10.0, -50.0])
pf.plot_local_explain_piecewise_linear_act(
    net, 
    explain_this, 
    n_samples=1_000,
    magnitude=True, 
    include_prediction=True, 
    include_bias=True, 
    fig_size=(6.4, 4.8), 
    no_zero_contributions=False, 
    ann=False, 
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_pos_pred")

# Empirical explanation
explain_this = explain_this.reshape(-1, p)
pf.plot_local_contribution_empirical(
    net, 
    explain_this, 
    sample=True, 
    median=True, 
    n_samples=1_000,
    variable_names=copy.deepcopy(variable_names), 
    class_names=["binary prediction"], 
    magnitude=True,
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_pos_pred_emp_approach")

explain_this = torch.tensor([-1.50, -1.50, 10.0, -50.0])
pf.plot_local_explain_piecewise_linear_act(
    net, 
    explain_this, 
    n_samples=1_000,
    magnitude=True, 
    include_prediction=True, 
    include_bias=True, 
    fig_size=(6.4, 4.8), 
    no_zero_contributions=False, 
    ann=False, 
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_net_pred")

# Empirical explanation
explain_this = explain_this.reshape(-1, p)
pf.plot_local_contribution_empirical(
    net, 
    explain_this, 
    sample=True, 
    median=True, 
    n_samples=1_000,
    variable_names=copy.deepcopy(variable_names), 
    class_names=["binary prediction"], 
    magnitude=True,
    save_path="implementations/local_explain/linear_problem/imgs/local_explain_net_pred_emp_approach")