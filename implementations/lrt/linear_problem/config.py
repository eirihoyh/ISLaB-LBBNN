#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NOTE: Used lambda min=-10 and lambda max=-7, but had high init prob of including painput params
config = {}
config['num_epochs' ] = 200  
config['lr'] = .1
config['post_train_epochs'] = 0  # If post training, then only MPM will be trained
config['patience'] = 10000
config['n_nets'] = 10  # Number of different networks to run for one problem
config['n_layers'] = 4+2 # Two hidden + (one input+one output)
config['hidden_dim'] = 20 # Number of hidden nodes per layer
config['num_transforms'] = 2  # For normalizing flows
config['class_problem'] = True  # If classification problem or not
config['inclusion_prob_prior'] = 0.001
config['std_prior'] = 2.5
config['lower_init_lambda'] = -10
config['upper_init_lambda'] = -7
config['high_init_covariate_prob'] = True  # If true, the inital covariate probs for the covariates will be set to lambda=5
# For the linear and non-linear problem
config['n_samples'] = 4*10**4
config['non_lin'] = False  # Wanting to create a non-linear or linear dataset 
config['test_samples'] = 5000

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = True  # If we should save the results