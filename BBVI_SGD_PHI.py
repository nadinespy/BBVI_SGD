#%% Variational Inference: blackbox variational inference with stochastic gradient descent.
#%%

from __future__ import absolute_import
from __future__ import print_function

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:58:41 2018

@author: nspychala
"""


import os
import scipy.io as sio
import math as m
import argparse
from functools import partial

import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score


import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
 
from autograd import grad

from joblib import Parallel, delayed
from math import sqrt

#%% Specify gradient descent optimization algorithms and model in variational inference.
#%%

def adam(grad, x, callback = None, num_iters_GD = None,
         step_size = None, b1 = 0.9, b2 = 0.999, eps = 10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters_GD):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate (direction).
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x

def vanilla_gradient_descent(grad, x, callback=None, num_iters_GD=150, step_size=0.001):
    """Does vanilla gradient descent Nadonski"""
    """(Not really vanilla: categories of optimization refer to how much data is used for cost function,""" 
    """with vanilla GD using all data, and other variants e. g. just parts of it)"""
    for i in range(num_iters_GD):
        g = grad(x, i)
        if callback: callback(x, i, g)
        x = x - step_size*g
    return x

def unpack_params(params):
    # Variational dist is a diagonal Gaussian.
    mean, log_std = params[:D], params[D:]
    return mean, log_std

def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""
 
    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std
 
    def gaussian_entropy(log_std):
        return  0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)
        
    rs = npr.RandomState(0)
    def variational_objective(params, t):                                       # variational_objective has as its input params, params is given in adam() via init_var_params, adam() gives init_var_params as input to gradient function of variational objective
        """Provides a stochastic estimate of the variational lower bound."""
        means, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + means            # sampling of log_sigma and mean (the latent variables) from a multivariate normal pdf that sums up to 1. 
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound
 
    gradient_function_of_variational_objective = grad(variational_objective)
 
    return variational_objective, gradient_function_of_variational_objective, unpack_params
 
#%% Specify an inference problem by its unnormalized log-density.
#%%

def log_density0(x, t):
    """Funnel function."""
    mu_posterior, log_sigma_posterior = x[:, 0], x[:, 1]
    log_sigma_posterior_log_density = norm.logpdf(log_sigma_posterior, 0, 1.35)
    mu_posterior_log_density = norm.logpdf(mu_posterior, 0, np.exp(log_sigma_posterior))
    return log_sigma_posterior_log_density + mu_posterior_log_density
    
#def log_density1(x, t):
#    """Two independent Gaussians."""
#    mu_posterior, log_sigma_posterior = x[:, 0], x[:, 1]
#    log_sigma_posterior_log_density = norm.logpdf(log_sigma_posterior, (-1), 1)
#    mu_posterior_log_density = norm.logpdf(mu_posterior, (-0.5), 1)
#    return log_sigma_posterior_log_density + mu_posterior_log_density

def log_density1(x, t):
    """One-dimensional Gaussian."""
    mu_posterior = x[:, 0]
    mu_posterior_log_density = norm.logpdf(mu_posterior, (-0.5), 1)
    return mu_posterior_log_density

def log_density2(x, t, corr = 0.0):
    """Two dependent Gaussians."""
    mu_posterior = [0, -1]
    std_posterior = [1, 0.8]
    covs = np.array([[std_posterior[0]**2,                      std_posterior[0]*std_posterior[1]*corr], 
                     [std_posterior[0]*std_posterior[1]*corr,   std_posterior[1]**2]])
    posterior_log_density = mvn.logpdf(x, mu_posterior, covs)
    return posterior_log_density
 
def log_density3(x, t, view=None, corr = [0.0, 0.0, 0.0]):
    """Three dependent Gaussians (sums up to 1)."""
    mu_posterior = np.array([0, -1, 0.5])
    std_posterior = [1, 0.8, 0.9]

    # Build correlation matrix.
    auto_corr = [1.0,1.0,1.0]
    corr_posterior = np.diag(auto_corr)
    corr_posterior[np.triu_indices(3, k=1)] = corr
    corr_posterior[np.tril_indices(3, k=-1)] = corr

    # Build covariance matrix.
    std_times_std_posterior = np.array([[std_posterior[0]**2,       std_posterior[0]*std_posterior[1],   std_posterior[0]*std_posterior[2]], 
                              [std_posterior[1]*std_posterior[0],   std_posterior[1]**2,                 std_posterior[1]*std_posterior[2]],
                              [std_posterior[2]*std_posterior[0],   std_posterior[2]*std_posterior[1],   std_posterior[2]**2]])
    cov_posterior = np.multiply(std_times_std_posterior,corr_posterior)

    if view is None:
        idx = [0,1,2]
    elif view == 0:
        idx = [0,2]
    elif view == 1:
        idx = [0,1]
    elif view == 2:
        idx = [1,2]
    posterior_log_density = mvn.logpdf(x, mu_posterior[idx], cov_posterior[idx][:,idx])

    return posterior_log_density
    
def log_density4(x, t, view = None, corr = [0.0, 0.0, 0.0]):
    """Funnel function for 3 variables, two of which depend on sigma."""
 
    if view is None:
        mu_posterior, third_parameter_posterior, log_sigma_posterior = x[:, 0], x[:, 1], x[:,2]
        log_sigma_posterior_log_density = norm.logpdf(log_sigma_posterior, 0, 1.35)
        mu_posterior_log_density = norm.logpdf(mu_posterior, 0, np.exp(log_sigma_posterior))
        third_parameter_posterior_log_density = norm.logpdf(third_parameter_posterior, 0.5, np.exp(log_sigma_posterior))
        posterior_log_density = log_sigma_posterior_log_density + mu_posterior_log_density + third_parameter_posterior_log_density
    elif view == 0:
        mu_posterior, log_sigma_posterior = x[:, 0], x[:, 1]                                                        # it is x[:, 0], x[:, 1] and not x[:, 0], x[:, 2], because of plot_isocontours
        mu_posterior_log_density = norm.logpdf(mu_posterior, 0, np.exp(log_sigma_posterior))
        log_sigma_posterior_log_density = norm.logpdf(log_sigma_posterior, 0, 1.35)
        posterior_log_density = log_sigma_posterior_log_density + mu_posterior_log_density
    elif view == 1:
        raise ValueError('geht net nadonksi')
    elif view == 2:
        third_parameter_posterior, log_sigma_posterior  = x[:, 0], x[:, 1]                                          # it is x[:, 0], x[:, 1] and not x[:, 1], x[:, 2], because of plot_isocontours
        log_sigma_posterior_log_density = norm.logpdf(log_sigma_posterior, 0, 1.35)
        third_parameter_posterior_log_density = norm.logpdf(third_parameter_posterior, 0.5, np.exp(log_sigma_posterior))
        posterior_log_density = log_sigma_posterior_log_density + third_parameter_posterior_log_density
        
    return posterior_log_density


#%% Set up plotting code.
#%%
    
def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    ax.contour(X, Y, Z)
        
def callback_3d_to_2d(params, t, g, all_params = None, unpack_params_specified=None, posterior_function = None):
    """Plots for each combination of two parameters the joint distribution, stores variational parameters."""
    
    print("Iteration {}".format(t))
    all_params.append(params) 

    fig, axes = plt.subplots(1,3, figsize=(8,3))
    mean, log_std = unpack_params_specified(params)

    try:
        target_distribution_1 = lambda x : np.exp(posterior_function(x, t, view=0))
        plot_isocontours(axes[0], target_distribution_1)        
        variational_contour_1 = lambda x: mvn.pdf(x, mean[[0,2]], np.diag(np.exp(2*log_std[[0,2]])))
        plot_isocontours(axes[0], variational_contour_1)
    except ValueError:
        fig.delaxes(axes[0])
        
    try:
        target_distribution_2 = lambda x : np.exp(posterior_function(x, t, view=2))
        plot_isocontours(axes[1], target_distribution_2)
        variational_contour2 = lambda x: mvn.pdf(x, mean[[1,2]], np.diag(np.exp(2*log_std[[1,2]])))
        plot_isocontours(axes[1], variational_contour2)
    except ValueError:
        fig.delaxes(axes[1])
            
    try:
        target_distribution_3 = lambda x : np.exp(posterior_function(x, t, view=1))
        plot_isocontours(axes[2], target_distribution_3)
        variational_contour3 = lambda x: mvn.pdf(x, mean[[0,1]], np.diag(np.exp(2*log_std[[0,1]])))
        plot_isocontours(axes[2], variational_contour3)
    except ValueError:
        fig.delaxes(axes[2])
    
    import PIL.ImageOps    
    from PIL import Image
    #image = Image.open('your_image.png')

    PIL.ImageOps.invert(fig)
    
    plt.draw()
    plt.pause(1.0/30.0)
    
def callback_2d(params, t, g, all_params = None, posterior_function = None, unpack_params_specified = None):
    """Plots joint distribution, stores variational parameters."""
    
    print("Iteration {}".format(t))
    all_params.append(params)
    
    fig, axes = plt.subplots(1,1, figsize=(5,3))
    target_distribution = lambda x : np.exp(posterior_function(x, t))
    plot_isocontours(axes, target_distribution)
 
    mean, log_std = unpack_params_specified(params)
    variational_contour = lambda x: mvn.pdf(x, mean, np.diag(np.exp(2*log_std)))
    plot_isocontours(axes, variational_contour)
    
    PIL.ImageOps.invert(fig)
    plt.draw()
    plt.pause(1.0/30.0)
    
def callback_no_plot(params, t, g, all_params = None, posterior_function = None, unpack_params_specified = None):
    """Does not plot, stores variational parameters."""
    all_params.append(params)
        
#%% Define variational inference procedure.
#%%

def run_one_chain(density_function, all_params, D, num_samples, callback_function, optimizer,**kwargs):
    """Does variational inference for a given target distribution."""
    specified_variational_objective, specified_gradient_function_of_variational_objective, unpack_params_specified = \
    black_box_variational_inference(density_function, D, num_samples)
    init_mean = np.random.uniform(-1,1) * np.ones(D)
    init_log_std = np.random.uniform(-2.75,-2.25) * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    optimizer(specified_gradient_function_of_variational_objective,init_var_params,
              callback = partial(callback_function, all_params = all_params,
                            posterior_function = density_function,
                            unpack_params_specified = unpack_params_specified), 
            **kwargs)
        
def run_many_chains(density_function, all_params, num_iters_VI = 1, D = 1, num_samples = 2000, 
                    callback_function = None, optimizer = adam,**kwargs): 
    """Does variational inference for a given target distribution multiple times."""
        
    all_variational_means1 = []
    all_variational_means2 = []
    all_variational_means3 = []
    all_variational_sigmas1 = []
    all_variational_sigmas2 = []
    all_variational_sigmas3 = []
        
    for q in range(num_iters_VI):
        all_params = []
        
        run_one_chain(density_function, all_params, D, num_samples, callback_function, optimizer, **kwargs)
            
        all_params = list(zip(*all_params))
        means = all_params[:D]
        log_std = all_params[D:]
            
        all_variational_means1.append(means[0])
        try:
            all_variational_means2.append(means[1])
        except:
            pass
        try:
            all_variational_means3.append(means[2])
        except:
            pass
        
        all_variational_sigmas1.append(log_std[0])
        try:
            all_variational_sigmas2.append(log_std[1])
        except:
            pass
        try:
            all_variational_sigmas3.append(log_std[2])
        except:
            pass

    all_variational_means1 = np.asarray(all_variational_means1).T
    all_variational_means2 = np.asarray(all_variational_means2).T
    all_variational_means3 = np.asarray(all_variational_means3).T
    all_variational_sigmas1 = np.asarray(all_variational_sigmas1).T
    all_variational_sigmas2 = np.asarray(all_variational_sigmas2).T
    all_variational_sigmas3 = np.asarray(all_variational_sigmas3).T
        
    if D == 3:
        all_params_VI = np.dstack((all_variational_means1,  all_variational_means2,  all_variational_means3, all_variational_sigmas1,  all_variational_sigmas2, all_variational_sigmas3))
    elif D == 2: 
        all_params_VI = np.dstack((all_variational_means1,  all_variational_means2, all_variational_sigmas1,  all_variational_sigmas2))
    elif D == 1:
        all_params_VI = np.dstack((all_variational_means1, all_variational_sigmas1))
    
     
    return all_params_VI

#%% Execute variational inference procedure.
#%%
        
if __name__ == '__main__':
    
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(121, frameon=False)
    ax2 = fig.add_subplot(122, frameon=False)

    plt.ion()
    plt.show(block=False)

    varParams = []
    all_varParams = run_many_chains(log_density1, varParams, num_iters_VI = 1000, D = 1, num_samples = 2000, 
                                    callback_function = callback_no_plot, optimizer = adam, num_iters_GD = 5000,  
                                    step_size = 0.001) 

    os.chdir('/mnt/525C77605C773E33/Nadine/MA_Philosophy/master_thesis/scripts_VIIIT/VI_matrices')
    sio.savemat('varParams_1_0.001_5000_log_density1_adam.mat', {'all_params_VI': all_varParams})

   


    
    
    
    
