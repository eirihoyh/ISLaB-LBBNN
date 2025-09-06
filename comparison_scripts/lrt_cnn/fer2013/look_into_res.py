import copy
import numpy as np
import pandas as pd
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
from lrt_cnn_net import BCNN
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



X_train1_original_csv = pd.read_csv("data/fer2013/X_train1.csv").pixels.values
X_train1_original = np.array(list(map(lambda x: x.split(" "), X_train1_original_csv))).astype(np.float32)/255.

X_train2_original_csv = pd.read_csv("data/fer2013/X_train2.csv").pixels.values
X_train2_original = np.array(list(map(lambda x: x.split(" "), X_train2_original_csv))).astype(np.float32)/255.

X_train_original = np.concatenate((X_train1_original, X_train2_original),0)

X_test_original_csv = pd.read_csv("data/fer2013/X_test.csv").pixels.values
X_test_original = np.array(list(map(lambda x: x.split(" "), X_test_original_csv))).astype(np.float32)/255.

y_train_original = np.loadtxt("data/fer2013/y_train.txt", delimiter=",")
y_test_original = np.loadtxt("data/fer2013/y_test.txt", delimiter=",")

n,p = X_train_original.shape
p1 = int(np.sqrt(p))
p2 = p1
print(n,p1,p2,dim)

channels = 1  # Gray scale images

if class_problem:
    n_classes = len(np.unique(y_train_original))
    multiclass = n_classes > 1
else: 
    n_classes = 1  # Just need to set it to something above zero
    multiclass = False

BATCH_SIZE = int((n)/10)
# TEST_BATCH_SIZE = int(n*0.10) # Would normally call this the "validation" part (will be used during training)
# VAL_BATCH_SIZE = int(n*0.10) # and this the "test" part (will be used after training)

TRAIN_SIZE = int((n))


NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

print(NUM_BATCHES)

# assert (TRAIN_SIZE % BATCH_SIZE) == 0


test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)


def evaluate(model, test_data, samples=100):
    model.eval()

    nll_median = 0
    nll_full = 0

    correct_median = 0
    correct_full = 0
    total = 0
    all_probs_median = []
    all_probs_full = []
    all_preds_median = []
    all_preds_full = []
    all_labels = []

    _x = test_data[:, :-1]
    _y = test_data[:, -1]

    n = _x.shape[0]

    with torch.no_grad():
        images, labels = _x.to(DEVICE),_y.type(torch.LongTensor).to(DEVICE)
        
        outputs_median = torch.zeros(samples, images.shape[0], n_classes).to(DEVICE)
        outputs_full = torch.zeros(samples, images.shape[0], n_classes).to(DEVICE)
        for i in range(samples):
            outputs_median[i] = model.forward(images, ensemble=False, sample=True, calculate_log_probs=True)
            outputs_full[i] = model.forward(images, ensemble=True, sample=True, calculate_log_probs=True)

        outputs_median = outputs_median.mean(0)
        all_probs_median.append(outputs_median.cpu())
        neg_log_lik_median = model.loss(outputs_median, labels)
        nll_median = neg_log_lik_median / n
        outputs_full = outputs_full.mean(0)
        all_probs_full.append(outputs_full.cpu())
        neg_log_lik_full = model.loss(outputs_full, labels)
        nll_full = neg_log_lik_full / n
        
        
        _, predicted_median = torch.max(outputs_median.data, 1)
        _, predicted_full = torch.max(outputs_full.data, 1)

        total += labels.size(0)
        correct_median += (predicted_median == labels).sum().item()
        correct_full += (predicted_full == labels).sum().item()

        all_preds_median.extend(predicted_median.cpu().numpy())
        all_preds_full.extend(predicted_full.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
    all_probs_median = torch.cat(all_probs_median)
    all_probs_full = torch.cat(all_probs_full)
    all_targets = np.array(all_labels)
    ece_median = pip_func.ece_score(np.exp(all_probs_median.numpy()), all_targets)
    ece_full = pip_func.ece_score(np.exp(all_probs_full.numpy()), all_targets)
    print(f"ECE median: {ece_median:.4f}, ECE full: {ece_full:.4f}")
    accuracy_median = 100 * correct_median / total
    accuracy_full = 100 * correct_full / total
    print(f"Test Accuracy median: {accuracy_median:.2f}%")
    print(f"Test Accuracy full: {accuracy_full:.2f}%")

    return nll_median, nll_full, accuracy_median, accuracy_full, ece_median, ece_full


results = []

for i in range(n_nets):  # Use the list from fer2013_lrt.py
    print(f"Testing net {i}")
    model = torch.load(f"comparison_scripts/lrt_cnn/fer2013/network/net{i}", weights_only=False)#,map_location=torch.device('cpu'))
    model.to(DEVICE)
    nll_median, nll_full, accuracy_median, accuracy_full, ece_median, ece_full = evaluate(model, test_dat)
    print(f"Loss median: {nll_median:.4f}, Loss full: {nll_full:.4f}")
    print(f"Accuracy median: {accuracy_median:.4f}, Accuracy full: {accuracy_full:.4f}")
    print(f"ECE median: {ece_median:.4f}, ECE full: {ece_full:.4f}")
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
        'nll_median': nll_median.item(),
        'nll_full': nll_full.item(),
        'accuracy_median': accuracy_median,
        'accuracy_full': accuracy_full,
        'ece_median': ece_median,
        'ece_full': ece_full,
        'nr_active_weights': nr_active_weights
    })

df = pd.DataFrame(results)
df.to_csv('comparison_scripts/lrt_cnn/fer2013/results/lrt_networks_fer2013_lr_0.005_results.csv', index=False)
