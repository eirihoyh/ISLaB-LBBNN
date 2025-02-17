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
from flow_net import BayesianNetwork
import torch.nn.functional as F

os.chdir(current_dir) # set the working directory back to this one 

# --------------fer2013----------------

HIDDEN_LAYERS = config['n_layers'] - 2 
n_nets = config['n_nets']
class_problem = config["class_problem"]

path = "implementations/flow/fer2013/"


X_test_original_csv = pd.read_csv("data/fer2013/X_test.csv").pixels.values
X_test_original = np.array(list(map(lambda x: x.split(" "), X_test_original_csv))).astype(np.float32)/255.
y_test_original = np.loadtxt("data/fer2013/y_test.txt", delimiter=",")

n,p = X_test_original.shape

if class_problem:
    n_classes = len(np.unique(y_test_original))
    multiclass = n_classes > 1
else: 
    n_classes = 1  # Just need to set it to something above zero
    multiclass = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dat = torch.tensor(np.column_stack((X_test_original,y_test_original)),dtype = torch.float32)

# test_res = []
# for i in range(n_nets):
#     net = torch.load(path+f"network/net{i}", weights_only=False,map_location=torch.device('cpu'))
#     for c in range(n_classes):
#         pf.plot_model_vision_image(net, train_data=test_dat[:,:-1], train_target=test_dat[:,-1], c=c, save_path=path+f"contribution_plots/net{i}_class{c}")
    # pf.save_metrics(net, path=path+f"results/net{i}")
#     ece_full, ece_median, nll_full, nll_median = pip_func.get_ece_score(net, test_dat, device, n_samples=100, n_classes=n_classes)
#     test_res.append([ece_full, ece_median, nll_full.cpu().detach().numpy(), nll_median.cpu().detach().numpy()])


# test_res = np.array(test_res)
# np.save(path+"results/ece", test_res)


test_res = np.load(path+"results/ece.npy", allow_pickle=True)
print("Order: ece_full, ece_median, nll_full, nll_median")
print(f"Median: {np.median(test_res,0)}")
print(f"Min:    {np.min(test_res,0)}")
print(f"Max:    {np.max(test_res,0)}")
print("")


res = {}
data = np.loadtxt(path+f"results/flow_class_skip_2_hidden_200_dim_10000_epochs_0.005_lr_fer2013_sigmoid_full.txt", delimiter=",")
data_median = np.loadtxt(path+f"results/flow_class_skip_2_hidden_200_dim_10000_epochs_0.005_lr_fer2013_sigmoid_median.txt", delimiter=",")
res["median_acc_full"] = np.median(data, 0)[0]
res["min_acc_full"] = np.min(data, 0)[0]
res["max_acc_full"] = np.max(data, 0)[0]

res["median_acc_median"] = np.median(data_median, 0)[0]
res["min_acc_median"] = np.min(data_median, 0)[0]
res["max_acc_median"] = np.max(data_median, 0)[0]

res["median_density"] = np.median(data, 0)[1]
res["min_density"] = np.min(data, 0)[1]
res["max_density"] = np.max(data, 0)[1]

res["median_used_weights"] = np.median(data_median, 0)[1]
res["min_used_weights"] = np.min(data_median, 0)[1]
res["max_used_weights"] = np.max(data_median, 0)[1]
print("")

for k in res.keys():
    print(f"{k}: {res[k]}")

print("")



res = {}
res["used_weights"] = []

for n in range(n_nets):
    median_info = np.load(path+f"results/net{n}_median.npy", allow_pickle=True).item()
    res["used_weights"].append(median_info["used_weights"])

print("USED WEIGHTS:")
m = (res["used_weights"])
print(f"median: {np.median(m)}")
print(f"min: {np.min(m)}")
print(f"max: {np.max(m)}\n")




res = {}
res["median_depth"] = []
res["max_width"] = []

for n in range(n_nets):
    median_info = np.load(path+f"results/net{n}_median.npy", allow_pickle=True).item()
    res["median_depth"].append(median_info["avg_path_length"])
    search = True
    i = 0
    while search:
        search = np.sum(median_info["include_inputs"][i]) == 0
        i += 1

    res["max_width"].append(HIDDEN_LAYERS-i+2)

print("Avg. depth")
print(np.median(res["median_depth"]))
print(np.min(res["median_depth"]))
print(np.max(res["median_depth"]))
print("")

print("Max depth")
print(np.median(res["max_width"]))
print(np.min(res["max_width"]))
print(np.max(res["max_width"]))
print("")

res = {}
for i in range(p):
    res[f"x{i+1}_inclution_rate"] = 0
    
    for n in range(n_nets):
        median_info = np.load(path+f"results/net{n}_median.npy", allow_pickle=True).item()
        #res[d][f"x{i+1}_inclution_rate"] += (median_info["expected_depth_input"][i]>0)
        count = 0
        for j in median_info["include_inputs"]:
            count += j[i]
        res[f"x{i+1}_inclution_rate"] += (count>0)
    res[f"x{i+1}_inclution_rate"] /= n_nets
    if res[f"x{i+1}_inclution_rate"] > 0:
        print(f"x{i+1}_inclution_rate: {res[f'x{i+1}_inclution_rate']}")

# for k in res.keys():
#     print(f"{k}: {res[k]}")
