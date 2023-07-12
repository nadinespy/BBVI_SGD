#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:46:28 2019

@author: nspychala
"""

import scipy.io as sio
import glob
import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import make_axes

    
#%% Plotting.
#%% log_density3: plot all variational chains in one 3D scatterplot. Do that for all variational inference matrices specified in list_of_log_density3.
fontsize1 = 22
fontsize2 = 16

import matplotlib
font = {'family' : 'normal',
        'size'   : 20}

def load_VI_mat_phi_vec(VI_matrix_path, phi_vector_path):
    all_params_VI = sio.loadmat(VI_matrix_path,squeeze_me=True)['all_params_VI']   
                                                                                # SGD iterations x VI iterations x number of variational means (MAPs)
#    size_of_VI_matrix = all_params_VI.shape
#    num_variational_means = int(size_of_VI_matrix[2]/2)
#    all_maxprob_VI = all_params_VI[:,:,0:num_variational_means]   
#    all_maxprob_VI = np.transpose(all_maxprob_VI, (2, 1, 0))                    # number of variational means (MAPs) x VI iterations x SGD iterations
    all_params_VI = np.transpose(all_params_VI, (2, 1, 0))
    
    try: 
        phiVector = sio.loadmat(phi_vector_path)['phiVector']                       # file is loaded as dictionary, choose phiVector from dictionary
        phiVector =  np.squeeze(phiVector)[1:]                                      # exclude first element from phiVector
    except: 
        phiVector = sio.loadmat(phi_vector_path)['mean_phi']                       # file is loaded as dictionary, choose phiVector from dictionary
        phiVector =  np.squeeze(phiVector)[1:]    
    
    return all_params_VI, phiVector    

matplotlib.rc('font', **font)
def plot_nicely(all_params_VI, all_params_GD, phiVector, num_iters_GD):
    
    fig = plt.figure(figsize=(21.2,9))
    plt.suptitle('\n evolution of variational parameters and Φ', fontdict={'fontsize':fontsize1})
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    for i in range(0,999,9):
        x = all_params_GD[0,i,:]
        y = all_params_GD[1,i,:]
        z = all_params_GD[2,i,:]
        time_idx = np.arange(all_params_GD.shape[2])
        mappable = ax.scatter(x, y, z, c=time_idx, cmap='viridis', marker='.', alpha=0.3, zorder=2)
    ax.set_xlabel('\nMAP feature 1', linespacing=3.1, fontdict={'fontsize':fontsize1})
    ax.set_ylabel('\nMAP feature 2', linespacing=3.1, fontdict={'fontsize':fontsize1})
    ax.set_zlabel('\nMAP feature 3', linespacing=3.1, fontdict={'fontsize':fontsize1})
    #ax.set_title('evolution of variational parameters and Φ\n', fontdict={'fontsize':fontsize1})
    cax, _ = make_axes(ax, location='right', fraction=0.05)
    ax.tick_params(axis='y', labelsize=fontsize2)
    ax.tick_params(axis='x', labelsize=fontsize2)
    ax.tick_params(axis='z', labelsize=fontsize2)
    steps = 0.5
    ax.set_xticks(np.arange(-1,1.5,steps))                             # Third dimension in all_params_VI is of size 500, we set location for labels within this set of numbers, i.e., within the range of 0-500.
    ax.set_xticklabels(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    ax.set_yticks(np.arange(-1,1.5,steps))                             # Third dimension in all_params_VI is of size 500, we set location for labels within this set of numbers, i.e., within the range of 0-500.
    ax.set_yticklabels(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    ax.set_zticks(np.arange(-1,1.5,steps))                             # Third dimension in all_params_VI is of size 500, we set location for labels within this set of numbers, i.e., within the range of 0-500.
    ax.set_zticklabels(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
       
    
    ax = fig.add_subplot(1, 2, 2)
    
    for i in range(0,999,9):
        ax.plot(all_params_VI[0,i,:], c='b', alpha=0.3)
        ax.plot(all_params_VI[1,i,:], c='r', alpha=0.3)
        ax.plot(all_params_VI[2,i,:], c='mediumpurple', alpha=0.3)
        ax.plot(all_params_VI[3,i,:], c='g', alpha=0.3)
        ax.plot(all_params_VI[4,i,:], c='gold', alpha=0.3)
        ax.plot(all_params_VI[5,i,:], c='darkgoldenrod', alpha=0.3)

    cc = ax.plot(all_params_VI[0,999,:], c='b', label='mean 1')
    vv = ax.plot(all_params_VI[1,999,:], c='r', label='mean 2')
    pp = ax.plot(all_params_VI[2,999,:], c='mediumpurple', label='mean 3')
    oo = ax.plot(all_params_VI[0,999,:], c='g', label='log std 1')
    ss = ax.plot(all_params_VI[1,999,:], c='gold', label='log std 2')
    uu = ax.plot(all_params_VI[2,999,:], c='darkgoldenrod', label='log std 3')
    
    moving_average = np.int(np.float(num_iters_GD/moving_average_index))                                         # adjust according to iteration length; 100 is good for num_iters_GD = 5000, 30 for num_iters_GD = 500 and 250; 40 for num_iters_GD = 1000; 50 for num_iters_GD = 2000; 160 for num_iters_GD = 12000
    phiVector_moving_average = np.array(range(0,len(phiVector),1))
    phiVector_moving_average = phiVector_moving_average.astype(np.float64)
    phiVector_moving_average.fill(np.nan) 
    phiVector = numpy.ma.array(phiVector, mask=np.isnan(phiVector))
    
    for i in range(len(phiVector)):
        if i < moving_average:
            number_of_numbers_in_sum = np.count_nonzero(~np.isnan(phiVector[i:i+moving_average]))
            phiVector_moving_average[i] = np.sum(phiVector[i:i+moving_average])/number_of_numbers_in_sum
        elif i > (len(phiVector)-moving_average):
            number_of_numbers_in_sum = np.count_nonzero(~np.isnan(phiVector[i-moving_average:i]))
            phiVector_moving_average[i] = np.sum(phiVector[i-moving_average:i])/number_of_numbers_in_sum
        else:    
            number_of_numbers_in_sum = np.count_nonzero(~np.isnan(phiVector[i-moving_average:i+moving_average]))
            phiVector_moving_average[i] = np.sum(phiVector[i-moving_average:i+moving_average])/number_of_numbers_in_sum #Originally, it was (moving_average*2), but we need to divide by the numbers that went into the calculation of the sum (which has ignored nan values).

    ax2 = ax.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Φ', color=color, fontdict={'fontsize':fontsize1})  # we already handled the x-label with ax1
    xx = ax2.plot(phiVector, color='grey', alpha=0.5, linewidth = 1, label='Φ')
    yy = ax2.plot(phiVector_moving_average, color='k', linewidth = 3, label='moving \naverage \nof Φ')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize2)
    #ax2.tick_params(axis='y', labelsize=fontsize2)
    #ax2.tick_params(axis='x', labelsize=fontsize2)
    ax2.set_ylabel('Φ', fontsize=fontsize1)
    
    # Do legend.
    aa = cc+vv+pp+oo+ss+uu+xx+yy
    labs = [l.get_label() for l in aa]
    axbox = ax.get_position()
    #legend = ax.legend(aa, labs, fontsize = 'xx-small', loc = (axbox.x0 - 0.05 , axbox.y0 - 0.12), facecolor = 'white', framealpha = 1)
    legend = ax.legend(aa, labs, fontsize = 'x-small', loc = (axbox.x0 + 0.575 , axbox.y0 - 0.13), facecolor = 'white', framealpha = 1)
    legend.set_zorder(200)
    
    ax.set_xlabel('iterations',fontdict={'fontsize':fontsize1})
    ax.set_ylabel('variational parameters', color = 'r', fontdict={'fontsize':fontsize1})
    ax.tick_params(axis='y', labelcolor='r')
    steps = int((num_iters_GD+1)/5)
    ax.set_xticks(np.arange(0,all_params_VI.shape[2]+1,steps))                             # Third dimension in all_params_VI is of size 500, we set location for labels within this set of numbers, i.e., within the range of 0-500.
    ax.set_xticklabels(np.concatenate((np.array([1]), np.array(np.arange(int(num_iters_GD/5),int(num_iters_GD+1),steps)))))
    ax.tick_params(axis='x', labelsize=fontsize2)    
    ax.tick_params(axis='y', labelsize=fontsize2)                                                                                     # We define labels for the specified locations (hence, number of locations and number of labels must match).

    #plt.tight_layout()
    #fig.subplots_adjust(hspace=0.2, right=0.85)
    plt.subplots_adjust(wspace = 0.45)
    
    cbar = plt.colorbar(mappable, cax=cax)
    mappable.set_clim(vmin=1, vmax=num_iters_GD)                                            # Why do I attribute set_clim to mappable and not to cbar?
    cbar.set_label('iterations', fontdict={'fontsize':fontsize1})
    cbar.ax.tick_params(labelsize=fontsize2) 
    return fig

def plot_all_variational_chains(list_of_matrices, list_of_phi):
    for i,w in zip(list_of_matrices, list_of_phi):
        all_params_VI, phiVector = load_VI_mat_phi_vec(i,w)
        all_params_VI = all_params_VI[:,:,0:num_iters_GD_index]
        phiVector = phiVector[0:num_iters_GD_index]
        num_iters_GD = all_params_VI.shape[2]
        if num_iters_GD == 2000:
            all_params_GD = all_params_VI[:,:,range(0,2000,4)]                            # Plot only part of each variational chain, i.e. each 2nd, 4th, 10th step such that the chain has 500 elements in total. (If all elements were plotted, the plots would not be distinguishable.)
        if num_iters_GD == 2500:
            all_params_GD = all_params_VI[:,:,range(0,2500,5)]       
        elif num_iters_GD == 1000: 
            all_params_GD = all_params_VI[:,:,range(0,1000,2)]
        elif num_iters_GD == 5000: 
            all_params_GD = all_params_VI[:,:,range(0,5000,10)]
        elif num_iters_GD == 8000:
            all_params_GD = all_params_VI[:,:,range(0,8000,16)]
        elif num_iters_GD == 12000:
            all_params_GD = all_params_VI[:,:,range(0,12000,24)]
        elif num_iters_GD == 25000:
            all_params_GD = all_params_VI[:,:,range(0,20000,40)]
        elif num_iters_GD == 20000:
            all_params_GD = all_params_VI[:,:,range(0,20000,40)]
        elif num_iters_GD <= 500:
            all_params_GD = all_params_VI
        fig = plot_nicely(all_params_VI, all_params_GD, phiVector, num_iters_GD)                                     # Mmh, it should plot all variational chains, it should be all_params_GD, and not all_params_VI...
        png_name = i.split('/')[-1:][0]
        fig.savefig('/mnt/525C77605C773E33/Nadine/MA_Philosophy/master_thesis/scripts_VIIIT/plots/'+'all_'+png_name[:-4]+'_all_variational_chains_3D_scatterplot_with_phi.png', dpi=300,
                    bbox_inches='tight')  
        plt.clf()
    return all_params_VI, all_params_GD, phiVector

     
# Check whether order is the same in both lists!     
list_of_log_density = sorted(glob.glob('/mnt/525C77605C773E33/Nadine/MA_Philosophy/master_thesis/scripts_VIIIT/VI_matrices/selected/*_3_0.001_25000_*'))
list_of_phi_log_density = sorted(glob.glob('/mnt/525C77605C773E33/Nadine/MA_Philosophy/master_thesis/scripts_VIIIT/phi_vectors_selected/*all_varParams**_3_0.001_25000_*'))


# Adjust according to step-size.
num_iters_GD_index = 20000
moving_average_index = 160

all_params_VI, all_params_GD, phiVector = plot_all_variational_chains(list_of_log_density, list_of_phi_log_density)


