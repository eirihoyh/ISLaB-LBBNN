import numpy as np
from scipy import stats
import copy

import torch
from torchmetrics import R2Score, MeanSquaredError



def create_data_unif(n,beta=[10,1,1,1,1], dep_level=0.5,classification=False, non_lin=False):
    """
    Create simple dataset with four independent variables (drawn uniformly from -10 to 10). 
    Can make x3 dependent on x1 by using the following function
        x3 = dep*x1 + (1-dep)*x3
    where dep is between 0 and 1.
    Can choose between linear and non linear function:
        linear: y = x1 + x2 + 10 + e
        non-linear: y = x1 + x2 + x1*x2 + x1**2 + x2**2 + 10 + e
    Can choose between classification problem and regression problem:
        classification: y = 0 if y < median(y), 1 else
        regression: y = y
    Can note that x3 and x4 is never used in the functions --> should be ignored in the model.
    """
    # Create data
    # np.random.seed(42)
    x0 = np.ones(n)
    x1 = np.random.uniform(-10,10,n)
    x2 = np.random.uniform(-10,10,n)
    x3 = np.random.uniform(-10,10,n)
    x4 = np.random.uniform(-10,10,n)

    x3 = dep_level*x1 + (1-dep_level)*x3  # make x3 dependent on x1

    if non_lin:
        y = beta[0] + beta[1]*x1 + beta[2]*x2 + beta[3]*x1**2 + beta[4]*x2**2 + x1*x2 # non-linear model
    else:
        y = beta[0] + beta[1]*x1 + beta[2]*x2
    rand0 = stats.norm.rvs(scale=0.01, size=n)
    # rand0 = stats.norm.rvs(scale=0.5, size=n)
    y += rand0
    if classification:
        y -= y.min()
        y /= y.max()
        # y = np.round(y)
        y = (y > np.median(y))*1


    return y, np.concatenate((np.array([x0]).T, np.array([x1]).T, np.array([x2]).T, np.array([x3]).T, np.array([x4]).T), axis=1)

def create_bsr_data(n, urange=[-3, 3], func=1):
    '''
    Functions used in the Bayesian Symbolic Regression paper.
    Compared the results from their paper with my approach 
    and found that the LBBNN networks are, in general, not as
    good at finding underlying functions, and will often diverge
    from the underlying function when extrapollating.
    '''
    l, u = urange[0], urange[1]
    # x1 = np.random.uniform(l,u,n)
    x1 = np.linspace(-1.5,1.5, n)
    x2 = np.random.uniform(l,u,n)
    
    if func==1:
        y = 2.5*x1**4 - 1.3*x1**3 + 0.5*x2**2 - 1.7*x2
    elif func==2:
        y = 8*x1**2 + 8*x2**3 - 15
    elif func==3:
        y = 0.2*x1**3 - .5*x1 + 0.5*x2**3 - 1.2*x2
    elif func==4:
        y = 1.5*np.exp(x1) + 5*np.cos(x2)
    elif func==5:
        y = 6*np.sin(x1)*np.cos(x2)
    elif func==6:
        y = 1.35*x1*x2 + 5.5*np.sin((x1-1)*(x2-1))

    elif func == 7:
        rand0 = stats.norm.rvs(scale=0.02, size=n)
        y = x1 + 0.3*np.sin(2*np.pi*(x1 + rand0)) + 0.3*np.sin(4*np.pi*(x1 + rand0)) + rand0
    elif func == 8:
        rand0 = stats.norm.rvs(scale=1., size=n)
        y = 10 * np.sin(2 * np.pi * (x1)) + rand0

    # rand0 = stats.norm.rvs(scale=0.02, size=n)
    # y += rand0

    return y, np.concatenate((np.array([x1]).T, np.array([x2]).T), axis=1)

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

def weight_matrices(net):
    '''
    Get all mean values in all the probability distributions.
    Will be stored as a list, where each element in the list
    are matrices where the columns represents the nodes in the preceding
    layer, and the rows represents the hidden nodes in the succeeding layer.
    In the Input skip-connection LBBNN model, we will have the following 
    dimension from one hidden layer to another hidden layer:
        (number of hidden nodes)x(number of hidden nodes + number of input variables)
    It should be noted that the last columns are always the input variables.
    '''
    n_hidden_layers = nr_hidden_layers(net)
    weight_matrices = []
    for name, param in net.named_parameters():
        for i in range(n_hidden_layers+1):
            if f'linears.{i}.weight_mu' in name:
                weight_matrices.append(copy.deepcopy(param.data))
    return weight_matrices

def weight_matrices_numpy(net, flow=False):
    '''
    Transform all tensor mu matrices to numpy arrays
    '''
    w = weight_matrices(net)
    for i in range(len(w)):
        w[i] = w[i].cpu().detach().numpy()

    if flow:
        z = z_matrices_numpy(net)
        for j in range(len(z)):
            w[j] *= z[j]
    
    return w

def z_matrices(net):
    '''
    The z-values related to the flow implementation
    '''
    n_hidden_layers = nr_hidden_layers(net)
    weight_matrices = []
    for name, param in net.named_parameters():
        for i in range(n_hidden_layers+1):
            if f'linears.{i}.q0_mean' in name:
                weight_matrices.append(copy.deepcopy(param.data))
    return weight_matrices

def z_matrices_numpy(net):
    '''
    Transform all tensor z matrices to numpy arrays
    '''
    z = z_matrices(net)
    for i in range(len(z)):
        z[i] = z[i].detach().numpy()

    return z

def weight_matrices_std(net):
    '''
    Get all std values in all the probability distributions.
    Will be stored as a list, where each element in the list
    are matrices where the columns represents the nodes in the preceding
    layer, and the rows represents the hidden nodes in the succeeding layer.
    In the Input skip-connection LBBNN model, we will have the following 
    dimension from one hidden layer to another hidden layer:
        (number of hidden nodes)x(number of hidden nodes + number of input variables)
    It should be noted that the last columns are always the input variables.
    '''
    n_hidden_layers = nr_hidden_layers(net)
    weight_matrices = []
    for name, param in net.named_parameters():
        for i in range(n_hidden_layers+1):
            if f'linears.{i}.weight_rho' in name:
                weight_matrices.append(copy.deepcopy(torch.log1p(torch.exp(param)).cpu().data))
    return weight_matrices

def weight_matrices_std_numpy(net):
    '''
    Transform all tensor standard deviation matrices to numpy arrays
    '''
    w = weight_matrices_std(net)
    for i in range(len(w)):
        w[i] = w[i].detach().numpy()

    return w

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
    alphas = {}
    for name, param in net.named_parameters():
        # if param.requires_grad:
        for i in range(n_hidden_layers+1):
            #if f"linears{i}.lambdal" in name:
            if f"linears.{i}.lambdal" in name:
            #if f"l{i+1}.lambdal" in name:
                alphas[i] = copy.deepcopy(1 / (1 + np.exp(-param.cpu().data)))

    return list(alphas.values())


def get_alphas_numpy(net):
    '''
    Transform all tensor alpha matrices to numpy arrays
    '''
    a = get_alphas(net)
    for i in range(len(a)):
        a[i] = a[i].detach().numpy()

    return a


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

def clean_alpha_class(net, threshold, class_in_focus=0, alpha_list=None):
    '''
    Gives all active paths to a given class(s).
    Can either give one class in the form as an integer
    or multiple in the form of a list of integers.
    '''
    if alpha_list==None:
        alpha_list = get_alphas(net)
    num_classes = alpha_list[-1].shape[0]
    remove_list = [True]*num_classes
    remove_list[class_in_focus]= False
    alpha_list[-1][remove_list, :] = 0
    return clean_alpha(net, threshold, alpha_list=alpha_list)

def get_active_weights(clean_alpha_list):
    """
    Gives postion of all active weights (all non-zero weights).
    This is used to map all active connections for the path plots,
    where we will have that the first position is the node that the 
    weight is going to, while the second postion is where the weight
    is comming from. 
    In the terms of the alpha/weight matrix, the first position is 
    the row of the matrix (where it is going to), and the second 
    posistion is the column of the matrix (where it is comming from). 
    
    This function will make sure that it is easy to create interpretable 
    graph plots, as we only need to draw lines from the second postion 
    to the first postion
    """
    active_path_dict = {}
    for ind, alpha in enumerate(clean_alpha_list):
        active_path_dict[ind] = alpha.nonzero()

    return list(active_path_dict.values())

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

def input_inclusion_prob(net, a=None):
    '''
    Computes the expected amount of active paths from an input to
    the output in its given position. E.g., the expected amount
    of active paths from Input1 from Hidden layer 2.
    This is done by computing the probability of all paths from
    an input postition.
    We assume independence between paths, although there are high
    correlation between many of the paths as they will merge together
    in multiple occasions.
    Gives probability of input going all the way to the output 
    from a given layer.
    THINK that this actually gives the expected number of nodes
    from an input possition (in all layers) to output.  
    This sums all possible paths for the input nodes to go all the
    way to the output. This will happen for all input nodes from each
    layer. 
    '''
    if a==None:
        a = get_alphas_numpy(net)
    length = len(a)
    p = a[0].shape[1]
    prob_paths = {}
    layer_names = create_layer_name_list(length+1)
    for name in layer_names[:-1]:
        for i in range(p):
            prob_paths[f"Prob I{i} from {name}"] = 0
    print(prob_paths)
    print(name)
    # Limit how many matrices that will be multiplied
    lims = np.arange(1, length, 1)[::-1]
    if len(lims)==0: # If we have a linear model
        for xi in range(p):
            prob_paths[f"Prob I{xi} from {name}"] = a[0][0][xi]
    else:
        for i, name in enumerate(layer_names[:-1]):        
            if i == lims[0]: # means that we are at last matrix
                probs = a[i][:,-p:].T
            else:
                
                probs = a[i][:,-p:].T
                count = 0
                while count < lims[i]:
                    count += 1
                    probs = probs@a[i+count][:,:-p].T
            for xi in range(p):
                prob_paths[f"Prob I{xi} from {name}"] = probs[xi][0]
    
    return prob_paths 

def expected_number_of_weights(net):
    '''
    Simply the sum over all alpha values. This assumes
    independence between layers. This is, however, not fully correct
    as we know weights will be removed if all connections to a given
    node is removed. This should be encountered for somehow in later
    versions of the code. 

    This will (most likely) mean that the expected amount of weights 
    will be proportional to the prior inclution probability as most 
    weights will be determined redundant after training, which again mean 
    that these weights will be set equal to the prior probability 
    distribution to get a little penalization as possbile.
    '''
    return sum([sum(list(a.flatten())) for a in get_alphas_numpy(net)])

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


    return np.mean(sum_dists), sum_dists

def prob_width(net, p):
    '''
    TODO: Check that it works as it should
    '''
    probs = input_inclusion_prob(net)

    probs_values = list(probs.values())
    pr_w = {}
    for i in range(p):
        pr_w[i] = np.min((np.sum(probs_values[i::p]),1))

    # exp_depth_net = np.sum(np.sum([probs_values[i::p] for i in range(p)],0)*np.arange(1,n_hidden+2,1)[::-1])/np.sum(probs_values)
    return pr_w


def get_weight_and_bias_std(net, alphas_numpy, threshold=0.5):
    '''
    Have that std represents the weights and biases 
    '''
    std_weigth = weight_matrices_std_numpy(net)
    for i in range(len(std_weigth)):
        std_weigth[i] *= (alphas_numpy[i] > threshold)*1.

    return std_weigth


def get_weight_and_bias(net, alphas_numpy, median=True, sample=False, threshold=0.5):
    weights = weight_matrices_numpy(net)
    std_weigth = weight_matrices_std_numpy(net)

    if sample:
        for i in range(len(weights)):
            std = std_weigth[i]
            weights[i] += np.random.normal(0,std)

    if median:
        for i in range(len(weights)):
            weights[i] *= (alphas_numpy[i] > threshold)*1.
    else:
        for i in range(len(weights)):
            include = np.random.binomial(1,alphas_numpy[i])*1.
            weights[i] *= include
            alphas_numpy[i] = copy.deepcopy(include)

    return weights, alphas_numpy

def relu_activation(input_data,weights):
    output_list = []
    out = np.array([[]])
    for i, w in enumerate(weights[:-1]):
        out = np.concatenate((out, input_data.detach().numpy()),1)
        out = out@w.T
        out = out*(out>0) # ReLU activation
        # print(out)
        output_list.append(out)
    
    # No activation in last layer (prediction/output layer)
    out = np.concatenate((out, input_data.detach().numpy()),1)
    out = out@weights[-1].T
    # print(out)
    output_list.append(out)

    return out, output_list

def no_activation(input_data,weights):
    output_list = []
    out = np.array([[]])
    for i, w in enumerate(weights[:-1]):
        out = np.concatenate((out, input_data.detach().numpy()),1)
        out = out@w.T
        output_list.append(out)
    
    # No activation in last layer (prediction/output layer)
    out = np.concatenate((out, input_data.detach().numpy()),1)
    out = out@weights[-1].T
    output_list.append(out)

    return out, output_list

def get_active_nodes(clean_alpha_list, output_list_c):
    active_nodes_alpha_list = [] # List of active nodes in each hidden layer (+ output layer) from alpha mat
    for i in range(len(clean_alpha_list)):
        active_nodes_alpha_list.append((np.sum(clean_alpha_list[i].detach().numpy(), 1) > 0)*1)
    
    active_nodes_list = [] # List of local active nodes (after ReLU activation) while also being active in the alpha mat
    for i in range(len(clean_alpha_list)-1):  # Does not check last layer as it is the output and it could be negative
        active_nodes_list.append((active_nodes_alpha_list[i]*output_list_c[i][0] > 0)*1)

    return np.array(active_nodes_list)

def find_active_weights(weights, active_nodes_list, clean_alpha_list, dim):
    active_weights = copy.deepcopy(weights) # Weigths used in the local active nodes
    length = len(active_weights)
    # active_weights[-1] = active_weights[-1][c:c+1,:]
    for i in range(length-1): 
        active_weights[i] = active_weights[i]*clean_alpha_list[i].detach().numpy() # Make sure that redundant connections is removed.
        active_weights[i] = np.array([active_weights[i][j,:]*active_nodes_list[i,j] for j in range(len(active_nodes_list[i]))]) # Make sure that only weights going into active nodes are used
        active_weights[i+1][:,:dim] = np.array([active_weights[i+1][:,j]*active_nodes_list[i,j] for j in range(len(active_nodes_list[i]))]).T  # Remove weights going out of an inactive node

    active_weights[-1] = active_weights[-1]*clean_alpha_list[-1].detach().numpy() # Make sure that redundant connections is removed.
    
    return active_weights

def local_explain_relu(net, input_data, threshold=0.5, median=True, sample=False, n_samples=1, verbose=False, quantiles=[0.025,0.975]):
    '''
    Gives local explainability for a given input when using a
    network initiated using ReLU activation.
    Note that we have only implemented a local explainability 
    model for the ReLU activation function. Main reason for this
    is that we know that we will get a linear model whenever we 
    are using ReLU, hence it is very simple to compute the 
    contribution from each input variable.
    The function works by finding all active nodes after ReLU 
    activation, then finding all active paths in the model in 
    general, and then exploting this information to compute all
    contribution from each input variable, one at-a-time.

    Returns a dictionary of dictionary, indicating the class we
    want to see contribution from, and how much each variable
    contribute in the given class.

    TODO: Make it more efficient by ignoring variables that is not 
    contributing to the given prediction.
    '''
    contributions = {}
    preds = []
    for n in range(n_samples):
        if verbose: print(f"Sample nr. {n}")
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]
        
        weights, alphas_numpy = get_weight_and_bias(net, alphas_numpy, median, sample, threshold) 

        # Get alpha matrices to torch to work for "clean_alpha_class" func
        alphas = copy.deepcopy(alphas_numpy)
        for i in range(len(alphas)):
            alphas[i] = torch.tensor(alphas[i])

        out, output_list = relu_activation(input_data, weights)
        if verbose: print(out) # "Predicted" values after sending data through network
        preds.append(out)
        contribution_classes = {}
        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c:c+1,:]  # Only include weights going into the class of interest
            clean_alpha_list = clean_alpha_class(net, threshold=0.5, class_in_focus=c, alpha_list=copy.deepcopy(alphas))
            clean_alpha_list[-1] = clean_alpha_list[-1][c:c+1,:]
            dim, p = clean_alpha_list[0].shape
            output_list_c = copy.deepcopy(output_list)
            
            active_nodes_list = get_active_nodes(clean_alpha_list, output_list_c)
            active_weights = find_active_weights(weights_class, active_nodes_list, clean_alpha_list, dim)
            
            
            pred_impact = {}
            for pi in range(p):
                explain_this_numpy = copy.deepcopy(input_data.detach().numpy())
                remove_list = [True]*p
                remove_list[pi] = False # focus on one input at-a-time
                explain_this_numpy[0,remove_list] = 0
                x = np.array([[]])
                for aw in active_weights:
                    x = np.concatenate((x, explain_this_numpy), 1)
                    x = x@aw.T           

                pred_impact[pi] = x[0,0]
        
            contribution_classes[c] = pred_impact
        contributions[n] = contribution_classes

    mean_contribution = {}
    cred_contribution = {}
    for c in range(nr_classes):
        mean_contribution[c] = {}
        cred_contribution[c] = {}
        for pi in range(p):
            values = np.zeros(n_samples)
            for s in range(n_samples):
                values[s] = contributions[s][c][pi]
            mean_contribution[c][pi] = np.mean(values)
            # TODO: In cases with a lot of zeros, and a few digits, one could get [0,0]. This would harm the current plotting function.
            cred_contribution[c][pi] = np.quantile(values, quantiles) # diff CI, uses 95\% as standard

    return mean_contribution, cred_contribution, np.array(preds)


def local_explain_relu_magnitude(net, input_data, threshold=0.5, median=True, sample=False, n_samples=1, verbose=False, quantiles=[0.025,0.975], include_potential_contribution=True):
    contributions = {}
    preds = []
    for n in range(n_samples):
        if verbose: print(f"Sample nr. {n}")
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]
        
        weights, alphas_numpy = get_weight_and_bias(net, alphas_numpy, median, sample, threshold) 

        # Get alpha matrices to torch to work for "clean_alpha_class" func
        alphas = copy.deepcopy(alphas_numpy)
        for i in range(len(alphas)):
            alphas[i] = torch.tensor(alphas[i])

        out, output_list = relu_activation(input_data, weights)
        if verbose: print(out) # "Predicted" values after sending data through network
        preds.append(out)
        contribution_classes = {}
        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c:c+1,:]  # Only include weights going into the class of interest
            clean_alpha_list = clean_alpha_class(net, threshold=0.5, class_in_focus=c, alpha_list=copy.deepcopy(alphas))
            clean_alpha_list[-1] = clean_alpha_list[-1][c:c+1,:]
            dim, p = clean_alpha_list[0].shape
            output_list_c = copy.deepcopy(output_list)
            
            active_nodes_list = get_active_nodes(clean_alpha_list, output_list_c)
            active_weights = find_active_weights(weights_class, active_nodes_list, clean_alpha_list, dim)
            
            
            pred_impact = {}
            for pi in range(p):
                explain_this_numpy = np.ones((1,p))
                remove_list = [True]*p
                remove_list[pi] = False # focus on one input at-a-time
                explain_this_numpy[0,remove_list] = 0
                x = np.array([[]])
                for aw in active_weights:
                    x = np.concatenate((x, explain_this_numpy), 1)
                    x = x@aw.T           

                if include_potential_contribution:
                    # if the input value is equal to zero, it gives opposite contribution to prediction
                    pred_impact[pi] = -1.*x[0,0] if input_data.detach().numpy()[0,pi] == 0 else x[0,0]
                else:
                    pred_impact[pi] = 0 if input_data.detach().numpy()[0,pi] == 0 else x[0,0]

            contribution_classes[c] = pred_impact
        contributions[n] = contribution_classes

    mean_contribution = {}
    cred_contribution = {}
    for c in range(nr_classes):
        mean_contribution[c] = {}
        cred_contribution[c] = {}
        for pi in range(p):
            values = np.zeros(n_samples)
            for s in range(n_samples):
                values[s] = contributions[s][c][pi]
            mean_contribution[c][pi] = np.mean(values)
            cred_contribution[c][pi] = np.quantile(values, quantiles) # diff CI, uses 95\% as standard

    return mean_contribution, cred_contribution, np.array(preds)

def local_explain_grad(net, data, p, n_samples=10_000, sample=True, ensemble=False):
    '''
    NOTE: This function is mainly meant for networks that utilizes 
    picewise linear functions, as this will give the exact contribution
    from each covariate for a given prediction. 
    '''
    explanation = torch.zeros((n_samples,p))
    for j in range(n_samples):
        
        explain_this = data[:-1].reshape(-1, p)
        explain_this.requires_grad = True
        net.zero_grad()
        output = net.forward_preact(explain_this, sample, ensemble)
        output_value = output[0,0]
        output_value.backward()

        gradients = explain_this.grad
        explanation[j] = gradients[0]


    # remove variables that does not contribute to the prediction at all
    expl = np.where(explain_this[0] != 0.0, explanation.cpu().detach().numpy(), 0)
    return expl

# def global_explain_grad(net, data ,p, n_samples=1_000, sample=True, ensemble=False):
#     for d in data:
#         local_explain_grad(net, d, p, n_samples, sample, ensemble)

def train(net, train_data, optimizer, batch_size, num_batches, p, DEVICE, nr_weights, multiclass=False, verbose=True, post_train=False):
    net.train()

    inds = np.arange(0,len(train_data),1)
    shuffle_data = np.random.choice(inds, size=len(train_data), replace=False)
    train_data = train_data[shuffle_data]
    old_batch = 0
    
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = train_data[old_batch: batch_size * batch,0:p]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        if multiclass:
            target = _y.type(torch.LongTensor).to(DEVICE)
        else:
            target = _y.to(DEVICE)
            target = target.unsqueeze(1).float()
                
        net.zero_grad()
        outputs = net(data, sample=True, post_train=post_train)
        negative_log_likelihood = net.loss(outputs, target) 
        loss = negative_log_likelihood + net.kl() / num_batches
        loss.backward()
        optimizer.step()

    if verbose:
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        print('density', expected_number_of_weights(net)/nr_weights) # This is over ALL weights, not just active paths
        print('')
    return negative_log_likelihood.item(), loss.item()

def val(net, val_data, DEVICE, multiclass=False, reg=False, verbose=True, post_train=False):
    '''
    NOTE: Will only validate using median model. 
    '''
    net.eval()
    with torch.no_grad():
        _x = val_data[:, :-1]
        _y = val_data[:, -1]
        data = _x.to(DEVICE)
        if multiclass:
            target = _y.type(torch.LongTensor).to(DEVICE)
        else:
            target = _y.to(DEVICE)
            target = target.unsqueeze(1).float()
        outputs = net(data, ensemble=False, calculate_log_probs=True, post_train=post_train)
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl()

        if reg:
            metric = R2Score()
            a = metric(outputs.T[0], target.T[0]).cpu().detach().numpy()
        else:
            if multiclass:
                output1 = outputs#.T.mean(0)
                class_pred = output1.max(1, keepdim=True)[1]
                a = class_pred.eq(target.view_as(class_pred)).sum().item() / len(target)
            else:
                class_pred = outputs.round().squeeze()
                a = np.mean((class_pred.cpu().detach().numpy() == target.cpu().detach().numpy().T[0]) * 1)
    
    alpha_clean = clean_alpha(net, threshold=0.5)
    density_median, used_weigths_median, _ = network_density_reduction(alpha_clean)
    if verbose:
        print(f'val_loss: {loss.item():.4f}, val_nll: {negative_log_likelihood.item():.4f}, val_ensemble: {a:.4f}, used_weights_median: {used_weigths_median}\n')

    return negative_log_likelihood.item(), loss.item(), a

def test_ensemble(net, test_data, DEVICE, SAMPLES, CLASSES=1, reg=True, verbose=True, post_train=False, multiclass=False):
    net.eval()
    metr = []
    metr_median = []
    density = []
    used_weights = []
    ensemble = []
    ensemble_median = []
    if reg:
        ensemble_r2 = []
        ensemble_r2_median = []
    with torch.no_grad():
        _x = test_data[:, :-1]
        _y = test_data[:, -1]
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        outputs = torch.zeros(SAMPLES, _x.shape[0], CLASSES).to(DEVICE)
        outputs_median = torch.zeros(SAMPLES, _x.shape[0], CLASSES).to(DEVICE)
        for i in range(SAMPLES):
            outputs[i] = net.forward(data, sample=True, ensemble=True, calculate_log_probs=True, post_train=post_train)
            outputs_median[i] = net.forward(data, sample=True, ensemble=False, calculate_log_probs=True, post_train=post_train)

        # Take the mean of the predictions 
        outputs_mean = outputs.mean(0)  
        outputs_median_mean = outputs_median.mean(0)
        # Get current density (median prob model) 
        alpha_clean = clean_alpha(net, threshold=0.5)
        density_median, used_weigths_median, _ = network_density_reduction(alpha_clean)
        density.append(density_median)
        used_weights.append(used_weigths_median)

        if reg:
            metric = R2Score()
            mse = MeanSquaredError().to(DEVICE)
            a_r2 = metric(outputs_mean.T[0], target).cpu().detach().numpy()
            a_median_r2 = metric(outputs_median_mean.T[0], target).cpu().detach().numpy()

            a_rmse = np.sqrt(mse(outputs_mean.T[0], target).cpu().detach().numpy())
            a_median_rmse = np.sqrt(mse(outputs_median_mean.T[0], target).cpu().detach().numpy())
        else:
            if multiclass:
                output1 = outputs_mean#.T.mean(0)
                class_pred = output1.max(1, keepdim=True)[1]
                a = class_pred.eq(target.view_as(class_pred)).sum().item() / len(target)

                output1_median = outputs_median_mean#.T.mean(0)
                class_pred_median = output1_median.max(1, keepdim=True)[1]
                a_median = class_pred_median.eq(target.view_as(class_pred_median)).sum().item() / len(target)
            else:
                output1 = outputs_mean.T.mean(0)
                class_pred = output1.round().squeeze()
                a = np.mean((class_pred.cpu().detach().numpy() == target.cpu().detach().numpy()) * 1)

                output1_median = outputs_median_mean.T.mean(0)
                class_pred_median = output1_median.round().squeeze()
                a_median = np.mean((class_pred_median.cpu().detach().numpy() == target.cpu().detach().numpy()) * 1)
                
        if reg:
            ensemble.append(a_rmse)
            ensemble_median.append(a_median_rmse)
            ensemble_r2.append(a_r2)
            ensemble_r2_median.append(a_median_r2)
        else:
            ensemble.append(a)
            ensemble_median.append(a_median)

        metr.append(np.mean(ensemble))
        metr.append(np.mean(density))
        metr_median.append(np.mean(ensemble_median))
        metr_median.append(np.mean(used_weights))
        if reg:
            metr.append(np.mean(ensemble_r2))
            metr_median.append(np.mean(ensemble_r2_median))

    if verbose:
        print(np.mean(density), 'density median')
        print(np.mean(used_weights), 'used weights median')
        print(np.mean(ensemble), 'ensemble full')
        print(np.mean(ensemble_median), 'ensemble median')
        

    return metr, metr_median
