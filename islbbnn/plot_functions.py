import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import torch
import copy
from graphviz import Digraph
import pipeline_functions as pip_func


def plot_whole_path_graph(alpha_list, all_connections, save_path, zero_index=False, show=False):
    dot = Digraph(f"All paths!")
    n_layers = len(alpha_list) + 1
    dim = alpha_list[0].shape[0]
    if zero_index:
        adder = 0
    else:
        adder = 1
    layer_list = pip_func.create_layer_name_list(n_layers)
    all_names = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            if f >= dim: # Check if we use input in Hidden layer
                from_node = f"I_{f-dim+adder}"
            else:
                from_node = f"{layer_list[layer_ind]}_{f+adder}"
                
            if from_node not in all_names:  # Add from node as a used name
                dot.node(from_node)
                all_names.append(from_node)

            if t >= dim and layer_ind+1 < n_layers: # TODO: consider removing if statement
                to_node = f"I_{t-dim+adder}"
            else:    
                to_node = f"{layer_list[layer_ind+1]}_{t+adder}"

            if to_node not in all_names:  # Add to node as a used name
                dot.node(to_node)
                all_names.append(to_node)

            # connect from_node to to_node; from_node --> to_node. Label with connection prob
            dot.edge(from_node, to_node, label=f"α={alpha_list[layer_ind][t][f]:.2f}")
        
    dot.node(f"All paths", shape="Msquare")
    # dot.edges(edges)
    dot.format = 'png' # save as PNG file
    dot.strict = True  # Remove duplicated lines
    # print(dot.source)
    dot.render(save_path, view=show)


def plot_whole_path_graph_weight(weight_list, all_connections, save_path, zero_index=False, show=True):
    dot = Digraph(f"All paths!")
    n_layers = len(weight_list) + 1
    dim = weight_list[0].shape[0]
    if zero_index:
        adder = 0
    else:
        adder = 1
    layer_list = pip_func.create_layer_name_list(n_layers)
    all_names = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            if f >= dim:
                from_node = f"I_{f-dim + adder}"
            else:
                from_node = f"{layer_list[layer_ind]}_{f+adder}"
                
            if from_node not in all_names:
                dot.node(from_node)
                all_names.append(from_node)

            if t >= dim and layer_ind+1 < n_layers:
                to_node = f"I_{t-dim+adder}"
            else:    
                to_node = f"{layer_list[layer_ind+1]}_{t+adder}"

            if to_node not in all_names:
                dot.node(to_node)
                all_names.append(to_node)

            dot.edge(from_node, to_node, label=f"w={weight_list[layer_ind][t][f]:.2f}")
        
    dot.node(f"All paths", shape="Msquare")
    # dot.edges(edges)
    dot.format = 'png' # save as PNG file
    dot.strict = True
    # print(dot.source)
    dot.render(save_path, view=show)


def run_path_graph(net, threshold=0.5, save_path="path_graphs/all_paths_input_skip", zero_index=False, show=True):
    # net = copy.deepcopy(net)
    alpha_list = pip_func.get_alphas(net)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    all_connections = pip_func.get_active_weights(clean_alpha_list)
    plot_whole_path_graph(alpha_list, all_connections, save_path=save_path, zero_index=zero_index, show=show)

def run_path_graph_weight(net, threshold=0.5, save_path="path_graphs/all_paths_input_skip", zero_index=False, show=True, flow=False):
    # net = copy.deepcopy(net)
    weight_list = pip_func.weight_matrices_numpy(net, flow=flow)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    all_connections = pip_func.get_active_weights(clean_alpha_list)
    plot_whole_path_graph_weight(weight_list, all_connections, save_path=save_path, zero_index=zero_index, show=show)


def plot_local_contribution_empirical(
        net, 
        data, 
        sample=True, 
        median=True, 
        n_samples=1, 
        include_bias=True, 
        save_path=None, 
        n_classes=1, 
        class_names=None, 
        variable_names=None, 
        quantiles=[0.025,0.975], 
        include_zero_means=True, 
        magnitude=False, 
        include_potential_contribution=False):
    '''
    Empirical local explaination model. This should be used for tabular data as 
    images usually has too many variables to get a good plot.

    TODO: Make sure that it is up to date regarding previous notebooks and 
        that contributions is computed using the gradient approach (much more efficient).
    '''
    variable_names = copy.deepcopy(variable_names)
    if magnitude:
        mean_contribution, cred_contribution, preds = pip_func.local_explain_relu_magnitude(net, data, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles, include_potential_contribution=include_potential_contribution)
    else:
        mean_contribution, cred_contribution, preds = pip_func.local_explain_relu(net, data, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles)
    if class_names == None:
        class_names = np.arange(n_classes)
    if variable_names == None:
        variable_names = np.arange(data.shape[-1])
        variable_names = list(variable_names.astype(str))
    variable_names.append("bias")
    variable_names = np.array(variable_names)

    preds_means = np.mean(preds,0)[0]
    for c in mean_contribution.keys():
        preds_errors = np.quantile(preds[:,:,c], quantiles)
        variable_names_class = copy.deepcopy(variable_names)
        # labels = np.array([str(k) for k in mean_contribution[c].keys()])
        means = np.array(list(mean_contribution[c].values()))
        errors = np.array(list(cred_contribution[c].values())) 

        if not include_bias:
            # labels = labels[:-1]
            means = means[:-1]
            errors = errors[:-1]
            variable_names_class = variable_names_class[:-1]
        
        if not include_zero_means:
            include = means != 0
            variable_names_class = variable_names_class[include]
            means = means[include]
            errors = errors[include]

        means = np.append(means, preds_means[c])
        errors = np.vstack([errors, preds_errors])
        for indx, err in enumerate(errors):
            if err[0] == 0 and err[1] == 0:
                err[0] = means[indx]
                err[1] = means[indx]
        top = errors[:,1]-means
        bottom = means-errors[:,0]
        variable_names_class = np.append(variable_names_class, "Prediction")

        fig, ax = plt.subplots()
        
        ax.bar(variable_names_class, means, yerr=(bottom, top), align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Contribution')
        ax.set_xticks(variable_names_class)
        ax.tick_params(axis='x', rotation=90)
        ax.set_title(f'Empirical explaination of {class_names[c]}')
        ax.grid()
        if save_path != None:
            plt.savefig(save_path+f"_class_{class_names[c]}")

        plt.show()

def plot_path_individual_classes(net, CLASSES, path="individual_classes"):
    for c in range(CLASSES):
        include_list = [True]*CLASSES
        include_list[c] = False
        a = pip_func.get_alphas(net)
        a[-1][include_list,:] = 0
        clean_a = pip_func.clean_alpha(net, 0.5, alpha_list=a)
        print("Used weights: ", sum([np.sum(ai.detach().numpy()) for ai in clean_a]))

        all_connections = pip_func.get_active_weights(clean_a)

        plot_whole_path_graph(a, all_connections, save_path=path + f"/class{c}", show=False)


def plot_model_vision_image(net, train_data, train_target, c=0, net_nr=0, threshold=0.5, thresh_w=0.0, save_path=None):
    '''
    NOTE: Works just for quadratic images atm, should probably generalize to prefered
            dim at a later point
    '''
    
    colors = ["white", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    
    clean_a = pip_func.clean_alpha_class(net, threshold,class_in_focus=c)
    p = int(clean_a[0].shape[1]**0.5)
    img_avg = np.zeros(p*p)

    w = pip_func.weight_matrices(net)[-1][c, -p*p:].detach().numpy()
    w = np.where(clean_a[-1][c,-p*p:].detach().numpy() == 1, w, 0)
    
    avg_c_img = train_data[train_target == c].mean(axis=0).reshape((p,p))

    fig, axs = plt.subplots(len(clean_a)+1, figsize=(10,10))
    
    for ind, ca in enumerate(clean_a):
        out = ca.shape[0]
        img_layer = np.zeros(p*p)
        for j in range(out):
            # img_layer += ca[j,-p:].detach().numpy()
            img_layer += np.where(np.abs(w) >= thresh_w, ca[j,-p*p:].detach().numpy(), 0)

        img_avg += img_layer
        axs[ind].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
        if np.sum(img_layer) > 0:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5)#, vmin=min_max*-1, vmax=min_max*1)
        else:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            
        fig.colorbar(im, ax=axs[ind])
        axs[ind].set_title(f"Class {c}, Layer {ind}")
        axs[ind].set_xticks([])
        axs[ind].set_yticks([])
        

    # min_max = max(np.concatenate((img_pos, img_neg*-1)))
    min_max = max(np.concatenate((img_avg, img_avg*-1)))

    
    axs[ind+1].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
    im = axs[ind+1].imshow(img_avg.reshape((p,p)), cmap=cmap, alpha=0.5, vmin=0, vmax=min_max*1)
    axs[ind+1].set_title(f"Net: {net_nr} all layers")
    axs[ind+1].set_xticks([])
    axs[ind+1].set_yticks([])
    fig.colorbar(im, ax=axs[ind+1])
    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

def plot_local_contribution_images_contribution_empirical(net, explain_this, n_classes=1, class_names=None, sample=True, median=True, n_samples=100, quantiles=[0.025,0.975], save_path=None):
    '''
    NOTE: Only works for ReLU based networks 
    '''
    _, cred_contribution, _ = pip_func.local_explain_relu(net, explain_this, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles)

    p = int(explain_this.shape[-1]**0.5)

    # make cred interval for pred based on n_samples (NOTE: softmax function)
    all_preds = []
    for _ in range(10_000):
        net.eval()
        preds = net.forward(explain_this, sample=True, ensemble=False).detach().cpu().numpy()[0]
        all_preds.append(np.exp(preds)/sum(np.exp(preds)))

    lower, upper = np.quantile(all_preds,[0.025,0.975], 0)

    if class_names == None:
        class_names = np.arange(n_classes)

    colors_025 = ["blue", "white", "red"]
    colors_975 = ["blue", "white", "red"]
    cmap_025 = mcolors.LinearSegmentedColormap.from_list("", colors_025)
    cmap_975 = mcolors.LinearSegmentedColormap.from_list("", colors_975)
    used_img = explain_this.reshape((p,p))
    for i in range(n_classes):
        explained_c = np.array(list(cred_contribution[i].values())[:-1])
        
        explained_025 = explained_c[:,0].reshape((p,p))
        explained_025 = np.where(abs(explained_025)>0, explained_025, np.nan)
        explained_975 = explained_c[:,1].reshape((p,p))
        explained_975 = np.where(abs(explained_975)>0, explained_975, np.nan)
        
        maxima_025 = np.nanmax(explained_025)
        minima_025 = np.nanmin(explained_025)

        maxima_975 = np.nanmax(explained_975)
        minima_975 = np.nanmin(explained_975)
        fig, axs = plt.subplots(1,2, figsize=(8,8))


        maxima = np.max([maxima_025, maxima_975,0])
        minima = np.min([minima_025, minima_975,0])
                
        axs[0].imshow(used_img, cmap="Greys", vmin=torch.min(used_img), vmax=torch.max(used_img)+0.5)
        axs[1].imshow(used_img, cmap="Greys", vmin=torch.min(used_img), vmax=torch.max(used_img)+0.5)
        norm_mean = TwoSlopeNorm(vmin=minima-0.001, vcenter=0, vmax=maxima+0.001)
        im = axs[0].imshow(explained_025, cmap=cmap_025, norm=norm_mean)
        cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
        cbar.ax.set_yscale('linear')

        norm_std = TwoSlopeNorm(vmin=minima-0.001, vcenter=0, vmax=maxima+0.001)
        im = axs[1].imshow(explained_975, cmap=cmap_975, norm=norm_std)
        cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.ax.set_yscale('linear')
        
        axs[0].set_title(f"0.025 quantile")
        axs[1].set_title(f"0.975 quantile")

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.suptitle(f"Local explain class: {class_names[i]}. Credibility interval: [{lower[i]:.4f}, {upper[i]:.4f}]")
        plt.tight_layout(rect=[0, 0.03, 1, 1.3])
        if save_path != None:
            plt.savefig(f"{save_path}/{class_names[i]}.png", bbox_inches='tight')
        plt.show()


def plot_local_contribution_images_contribution_empirical_magnitude(net, explain_this, n_classes=1, class_names=None, sample=True, median=True, n_samples=100, quantiles=[0.025,0.975], save_path=None, include_potential_contribution=False):
    '''
    NOTE: Only works for ReLU based networks 
    '''
    _, cred_contribution, _ = pip_func.local_explain_relu_magnitude(net, explain_this, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles, include_potential_contribution=include_potential_contribution)

    p = int(explain_this.shape[-1]**0.5)

    # make cred interval for pred based on n_samples (NOTE: softmax function)
    all_preds = []
    for _ in range(10_000):
        net.eval()
        preds = net.forward(explain_this, sample=True, ensemble=False).detach().cpu().numpy()[0]
        all_preds.append(np.exp(preds)/sum(np.exp(preds)))

    lower, upper = np.quantile(all_preds,[0.025,0.975], 0)

    if class_names == None:
        class_names = np.arange(n_classes)

    colors_025 = ["blue", "white", "red"]
    colors_975 = ["blue", "white", "red"]
    cmap_025 = mcolors.LinearSegmentedColormap.from_list("", colors_025)
    cmap_975 = mcolors.LinearSegmentedColormap.from_list("", colors_975)
    used_img = explain_this.reshape((p,p))
    for i in range(n_classes):
        explained_c = np.array(list(cred_contribution[i].values())[:-1])
        
        explained_025 = explained_c[:,0].reshape((p,p))
        explained_025 = np.where(abs(explained_025)>0, explained_025, np.nan)
        explained_975 = explained_c[:,1].reshape((p,p))
        explained_975 = np.where(abs(explained_975)>0, explained_975, np.nan)
        
        maxima_025 = np.nanmax(explained_025)
        minima_025 = np.nanmin(explained_025)

        maxima_975 = np.nanmax(explained_975)
        minima_975 = np.nanmin(explained_975)
        fig, axs = plt.subplots(1,2, figsize=(8,8))


        maxima = np.max([maxima_025, maxima_975,0])
        minima = np.min([minima_025, minima_975,0])
                
        axs[0].imshow(used_img, cmap="Greys", vmin=torch.min(used_img), vmax=torch.max(used_img)+0.5)
        axs[1].imshow(used_img, cmap="Greys", vmin=torch.min(used_img), vmax=torch.max(used_img)+0.5)
        norm_mean = TwoSlopeNorm(vmin=minima-0.001, vcenter=0, vmax=maxima+0.001)
        im = axs[0].imshow(explained_025, cmap=cmap_025, norm=norm_mean)
        cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
        cbar.ax.set_yscale('linear')

        norm_std = TwoSlopeNorm(vmin=minima-0.001, vcenter=0, vmax=maxima+0.001)
        im = axs[1].imshow(explained_975, cmap=cmap_975, norm=norm_std)
        cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.ax.set_yscale('linear')
        
        axs[0].set_title(f"0.025 quantile")
        axs[1].set_title(f"0.975 quantile")

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.suptitle(f"Local explain class: {class_names[i]}. Credibility interval: [{lower[i]:.4f}, {upper[i]:.4f}]")
        plt.tight_layout(rect=[0, 0.03, 1, 1.3])
        if save_path != None:
            plt.savefig(f"{save_path}/{class_names[i]}.png", bbox_inches='tight')
        plt.show()

def get_metrics(net, threshold=0.5):
    net = copy.deepcopy(net)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = pip_func.create_layer_name_list()

    density, used_weights, tot_weights = pip_func.network_density_reduction(clean_alpha_list)
    print(f"Used {used_weights} out of {tot_weights} weights in median model.")
    print(f"Density reduced to:\n{(density*100):.4f}%\n")

    expected_density = pip_func.expected_number_of_weights(net)
    print(f"Expected to use {expected_density:.2f} out of {tot_weights} weights in the full model.")
    print(f"Density reduced to:\n{((expected_density/tot_weights)*100):.4f}%\n")

    mean_path_length, length_list = pip_func.average_path_length(clean_alpha_list)
    print(f"Average path length in network:\n{mean_path_length:.2f}")
    include_inputs = pip_func.include_input_from_layer(clean_alpha_list)
    print("Following inputs have been included:")
    for i, include in enumerate(include_inputs):
        print(f"Layer {layer_names[i]}: {include}\t  -->\tNr of inputs used: {sum(include)}")

    prob_include_input = pip_func.input_inclusion_prob(net)
    print("\nExpected number of input nodes included from a given layer:")
    for i,j in zip(prob_include_input.keys(), prob_include_input.values()):
        print(f"{i} --> {j:.4f}")

    print(pip_func.prob_width(net, p))


def save_metrics(net, threshold=0.5, path="results/all_metrics"):
    # net = copy.deepcopy(net)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = pip_func.create_layer_name_list()

    density, used_weights, tot_weights = pip_func.network_density_reduction(clean_alpha_list)
    mean_path_length, length_list = pip_func.average_path_length(clean_alpha_list)
    include_inputs = pip_func.include_input_from_layer(clean_alpha_list)

    metrics_median = {}
    metrics_median["layer_names"] = layer_names
    metrics_median["tot_weights"] = tot_weights
    metrics_median["used_weights"] = used_weights.detach().numpy()
    metrics_median["density"] = density.detach().numpy()
    metrics_median["avg_path_length"] = mean_path_length
    metrics_median["include_inputs"] = include_inputs
    
    
    # Save median model dictionary using numpy
    np.save(path+"_median", metrics_median)


    expected_density = pip_func.expected_number_of_weights(net)
    prob_include_input = pip_func.input_inclusion_prob(net)

    metrics_full = {}
    metrics_full["layer_names"] = layer_names
    metrics_full["tot_weights"] = tot_weights
    metrics_full["expected_nr_of_weights"] = expected_density
    metrics_full["density"] = expected_density/tot_weights
    metrics_full["expected_nr"] = prob_include_input
    metrics_full["width_prob"] = pip_func.prob_width(net, p)
    
    # Save full model dictionary using numpy
    np.save(path+"_full", metrics_full)

