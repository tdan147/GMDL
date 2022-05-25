#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import copy
import numpy as np
from pylab import plt



def plot_vec(start,vec,*args,**kwargs):
    plt.plot([start[0],start[0]+vec[0]],
             [start[1],start[1]+vec[1]],*args,**kwargs)
def plot_ellipse_using_cholesky(center,Sigma,scale=1,*args,**kwargs): 
    A = np.linalg.cholesky(Sigma).T
    theta = np.linspace(0,2*np.pi,100)
    c = np.cos(theta)
    s = np.sin(theta)
    circle = np.array([c,s])
    ellipse=scale*A.dot(circle)
    plt.plot(center[0]+ellipse[0],center[1]+ellipse[1],'-',*args,**kwargs)
def plot_cov_axes(Sigma,scale=1,*args,**kwargs):
    eigvals,eigvecs=np.linalg.eigh(Sigma)
    # The function output are sorted from small to large.
    # So flip it. 
    eigvals=eigvals[::-1]
    eigvecs=eigvecs[:,::-1]
    v1,v2=eigvecs.T
    s1,s2=np.sqrt(eigvals)    
    plot_vec(mu,eigvecs[:,0]*scale*np.sqrt(eigvals[0]),color='k',*args,**kwargs)
    plot_vec(mu,eigvecs[:,1]*scale*np.sqrt(eigvals[1]),color='k',*args,**kwargs)

def plot_data(x,*args,**kwargs):
    kwargs = copy.deepcopy(kwargs)
    if 'color' not in kwargs:
        kwargs['color']='m'
    plt.plot(x[0],x[1],'o',*args,**kwargs)

def my_axes(b=15):     
    plt.axis('scaled')
#    plt.xlim(-b,b)
#    plt.ylim(-b,b)  
    plt.grid('on')

def plt_pi(pi,colors):
    K = len(pi)
    labels=[r'$\pi_{0}$'.format(k) for k in range(1,K+1)]
    pos =np.arange(1,K+1)
    for k,pi_k in enumerate(pi):
        color=colors[k]
        plt.bar(k+1, pi_k,align='center',color=color)           
        if K<8:
            plt.xticks(pos, labels)
        else:
            plt.xticks([])
        
    plt.ylim(0,1)   
def plt_counts(Ns,colors):
    K=len(Ns)
    labels=[r'$N_{0}$'.format(k) for k in range(1,K+1)]
    pos =np.arange(1,K+1) 
    for k,N_k in enumerate(Ns):
        color=colors[k]
        plt.bar(k+1, N_k,align='center',color=color)       
        if K<8:
            plt.xticks(pos, labels)
        else:
            plt.xticks([])
def plt_normalized_counts(Ns,colors):
    K = len(Ns)
    N=Ns.sum()
    labels=[r'$\pi_{0}$'.format(k) for k in range(1,K+1)]
    pos =np.arange(1,K+1)
    for k,N_k in enumerate(Ns):
        color=colors[k]
        plt.bar(k+1, float(N_k)/N,align='center',color=color)       
        if K<8:
            plt.xticks(pos, labels)
        else:
            plt.xticks([])

    plt.ylim(0,1)
