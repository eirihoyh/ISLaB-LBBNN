from scipy import stats
import numpy as np
import torch
from sklearn.metrics import mean_pinball_loss
n_samples = 80000

def create_data_unif(n=n_samples,beta=[100,1,1,1,1], dep_level=0.5,classification=False, non_lin=True):
    # Create data
    np.random.seed(47)
    x1 = np.random.uniform(-10,10,n)
    x2 = np.random.uniform(-10,10 ,n)
    x3 = np.random.uniform(-10,10,n)
    x4 = np.random.uniform(-10,10 ,n)

    x3 = dep_level*x1 + (1-dep_level)*x3  # make x3 dependent on x1

    if non_lin:
        y = beta[0] + beta[1]*x1 + beta[2]*x2 + beta[3]*x1**2 + beta[4]*x2**2 + x1*x2 # non-linear model
    else:
        y = beta[0] + beta[1]*x1 + beta[2]*x2
    
    rand0 = stats.norm.rvs(scale=0.01, size=n)
    y += rand0
    if classification:
        y -= y.min()
        y /= y.max()
        # y = np.round(y)
        y = (y > np.median(y))*1

    
    return y, np.concatenate((np.array([x1]).T, np.array([x2]).T, np.array([x3]).T, np.array([x4]).T), axis=1)


def nr_hidden_layers(net):
    """
    Brute force way of finding the number of hidden layers.
    It is also very dependent on having the name of the layers
    being "linears", so should consider making it more genereal 
    future implementations.
    """
    last_name = "linears.0."
    for name, _ in net.named_parameters():
        last_name = name
    return int(last_name[8])
    
def get_alphas(net):
    """
    Get all weight probabilities in the model.
    Will be stored as a list, where each element in the list
    are matrices where the columns represents the nodes in the preceding
    layer, and the rows represents the hidden nodes in the succeeding layer.
    In the Input skip-connection LBBNN model, we will have the following 
    dimension from one hidden layer to another hidden layer:
        (number of hidden nodes)x(number of hidden nodes + number of input variables)
    It should be noted that the last columns are always the input variables.
    """
    n_hidden_layers = nr_hidden_layers(net)
    weights = {}
    for name, param in net.named_parameters():
        # if param.requires_grad:
        for i in range(n_hidden_layers+1):
            if f"linears.{i}.weight" in name:
                weights[i] = param.cpu().data
            
    return list(weights.values())


def clean_alpha(net, threshold, alpha_list=None):
    '''
    Removes all non-active paths from the alpha matrices.
    An active path is defined as weights that connects an input variable to an output 
    node.
    For instance 
        x --> w --> Non-lin-transform --> output
    is an active path, while 
        x --> w --> Non-lin-transform     output
        x     w --> Non-lin-transform --> output
    are not active paths.
    An input variable can have multiple active paths, both from the same layer, and 
    from different layers.
    Works by setting all alpha > threshold equal to 1. Then, from the output layer, 
    we set all rows in the preceeding alpha matrix (e.g. alpha matrix for second to 
    last and last hidden layer) equal to zero if the sum of the corresponding column 
    in the succeeding alpha matrix (e.g. alpha matrix for last hidden layer and output
    layer) is equal to zero (i.e. no connection from output to that hidden node). This 
    is done for all pairs of alpha matrices, all the way to the input layer. Then we 
    will go from the input layer, and set the succeeding rows equal to zero if the 
    corresponding column in the preceeding alpha matrix sums to zero. This is done all 
    the way to the output layer, and will remove all weights that is not connected to an 
    input variable, and not connected to the output. Doing this process will give matrices 
    with only active paths that goes from an input variable, to an output node.
    This function was originally ment for median probability models (threshold=0.5), but 
    can work for any preffered threshold. 
    NOTE: The alpha list should go from input layer to output layer
    '''
    if alpha_list==None:
        alpha_list = get_alphas(net)
    dim = alpha_list[0].shape[0] # NOTE: we assume same amount of nodes in each layer 
    clean_dict = {}
    for ind, alpha in enumerate(alpha_list):
        clean_dict[ind] = (alpha > threshold)*1
    for ind in np.arange(1, len(alpha_list))[::-1]:
        clean_dict[ind-1] = (clean_dict[ind-1].T*(sum(clean_dict[ind][:,:dim])>0)).T*1
    for ind in np.arange(1,len(alpha_list)):
        clean_dict[ind] = torch.cat(((clean_dict[ind][:,:dim]*(sum(clean_dict[ind-1].T)>0))*1, clean_dict[ind][:,dim:]), 1)

    return list(clean_dict.values())



def include_input_from_layer(clean_alpha_list):
    '''
    Find what layers the inputs are expected to come from.
    This will then be from the cleaned version, where we have
    already decided a threshold 

    Returns a True/False statement for all inputs from the different
    layers
    '''
    p = clean_alpha_list[0].shape[1]
    include_list = []
    for alpha in clean_alpha_list:
        include_input = np.sum(alpha[:,-p:].detach().numpy(), 0) > 0
        include_list.append(include_input)

    return include_list



def create_layer_name_list(n_layers=None, net=None):
    """
    Get names for all layers based on the total amount of layers.

    TODO: Make more general in the sense that we should be 
        able to tackle this by either giving the number of hidden
        layers, or use a network to compute the number of hidden
        layers.
    """
    if net != None:
        n_layers = nr_hidden_layers(net) + 2
    
    layers = ["I"]

    for layer in range(n_layers-2):
        layers.append(f"H{layer+1}")

    layers.append("Output")
    return layers

def network_density_reduction(clean_alpha_list):
    """
    Computes the number of used weights, total amount of weights, and the relative 
    reducation of weights in a meidan probability LBBNN model. Need to give a list 
    of clean alpha matrices (matrices with only active paths).
    """
    used_weights = 0
    tot_weights = 0
    for a in clean_alpha_list:
        shape_a = a.shape
        used_weights += sum(sum(a))
        tot_weights += shape_a[0]*shape_a[1]

    return used_weights/tot_weights, used_weights, tot_weights


def average_path_length(clean_alpha_list):
    """
    Computes the average path length for a given input and in total
    for all inputs.
    TODO: Go through function and check that it actually works as 
            it should...
    """
    length_list = len(clean_alpha_list)
    p = clean_alpha_list[0].shape[1]
    sum_dists = np.array([])
    # Check from intput to output
    for i in range(length_list):
        for xi in range(p):
            path_length = clean_alpha_list[i][:,-(xi+1)].detach().numpy()*(length_list-i)
            path_length = path_length[path_length!=0]
            sum_dists = np.concatenate((sum_dists, path_length))
    
    # Check if hidden node have expanded
    #for i in range(1, length_list,1):
    #    curr_alpha = clean_alpha_list[i]
    #    for dimi in range(curr_alpha.shape[1]-p):
    #        incoming_weights = np.sum(clean_alpha_list[i-1][dimi,:].detach().numpy())
    #        extra_weights = np.sum(curr_alpha[:,dimi].detach().numpy()) - 1
    #        path_length = np.array([1]*extra_weights*incoming_weights)*(length_list-i+1) 
    #        path_length = path_length[path_length>0]
    #        sum_dists = np.append(sum_dists, path_length)


    return np.mean(sum_dists), sum_dists



def ece_score_binary(output, label, n_bins=10):

    pypy = np.array(output)
    y_test = np.array(label,dtype = int)
   
    py = np.zeros((pypy.shape[0],2))
    for i, prob in enumerate(pypy.squeeze()): #need two dimension for predicted probs
        py[i,1] = prob
        py[i,0] = 1 - prob
          
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
   

    py_index = np.argmax(py, axis=1)
  
   
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


def ece_score(output, label, n_bins=10):

    py = np.array(output)
    y_test = np.array(label,dtype = int)
     
    

    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
   

    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    
   
    
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


def pinball_loss(y_true,y_pred):
    alpha = np.arange(0.05,1.00,0.05) #from 0.05 -> 0.95 in 0.05 increments
    loss = np.zeros(len(alpha))
    for i,a in enumerate(alpha):
        loss[i] = mean_pinball_loss(y_true, y_pred,alpha = a)

    return loss.mean()

