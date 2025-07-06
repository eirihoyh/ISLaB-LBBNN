#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NOTE: Used lambda min=-3 and lambda max=-1
config = {}
config['num_epochs' ] = 100 #250  
config['lr'] = .01
config['post_train_epochs'] = 0  # If post training, then only MPM will be trained
config['patience'] = 10000
config['n_nets'] = 10  # Number of different networks to run for one problem
config['n_layers'] = 2+2 # Two hidden + (one input+one output)
config['hidden_dim'] = 256 # Number of hidden nodes per layer
config['num_transforms'] = 2  # For normalizing flows
config['class_problem'] = True  # If classification problem or not
config['inclusion_prob_prior'] = 0.1
config['std_prior'] = 1.
config['lower_init_lambda'] = 5
config['upper_init_lambda'] = 10
config['high_init_covariate_prob'] = True # If true, the inital covariate probs for the covariates will be set to lambda=5

# CNN specific
config['out_channels_list'] = [32,48,64]
config['kernel_size'] = 3
config['stride'] = 2
config['padding'] = 0



config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = True  # If we should save the results