#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""

import numpy as np
from pylab import plt
from GaussianInference import GaussianInference

from gmm_plt import *




def main(N,nu,kappa):  
    A = np.array([[1,-.5],[0,1]]).astype(np.float)
    Sigma = A.dot(np.eye(2)).dot(A.T)

    if nu != kappa:
        raise NotImplementedError(nu,kappa)
   
    name='N_{0:06}_pseudocounts_{1:06}'.format(N,nu)
    mu = np.array([2,4])
    np.random.seed(2)
    x = np.random.multivariate_normal(mean=mu,cov=Sigma,size=N)
    x = x.T # x is now 2 by N
    
    fig=plt.figure(name)

    plt.clf()
    
    plt.subplot(231)
#    plt.subplot(221)
    plot_data(x,ms=3)
    #plot_cov_axes(Sigma,scale=5,lw=2)
    plot_ellipse_using_cholesky(center=mu,Sigma=Sigma,scale=3,color='b',lw=1)
    my_axes()
    plt.title('data and GT')
       
 
    Psi=np.eye(2)
    m = np.zeros(2)

    
    gaussian_inference = GaussianInference(Psi=Psi,m=m,kappa=kappa,nu=nu)
    
    T1,T2=gaussian_inference.calc_suff_stats(x)
    mu_mle,Sigma_mle = gaussian_inference.calc_mle(T1,T2,N=N)
        
        
    plt.subplot(232)
#    plt.subplot(222)
    plot_data(x,ms=3)
    plot_ellipse_using_cholesky(center=mu,Sigma=Sigma,scale=3,color='b',lw=1)
    plot_ellipse_using_cholesky(mu_mle,Sigma=Sigma_mle,scale=3,color='r',lw=1)
    plt.legend(['data','GT','MLE'],loc='lower left',fontsize=10)
    my_axes() 
    plt.title('GT/MLE') 
          
    
    
    
    

    
    
    nSamples=10
    gaussian_inference.calc_posterior_hyper_params(N=N,T1=T1,T2=T2)
    
    np.random.seed(0)
    samples_from_the_iw_prior =  gaussian_inference.sample_from_the_iw_prior(nSamples)
    samples_from_the_niw_prior =  gaussian_inference.sample_from_the_niw_prior(nSamples)


    np.random.seed(0)
    samples_from_the_iw_posterior =  gaussian_inference.sample_from_the_iw_posterior(nSamples)
    np.random.seed(0)
    samples_from_the_niw_posterior=gaussian_inference.sample_from_the_niw_posterior(nSamples)
    
    plt.subplot(233)
#    plt.subplot(223)
    
    for i in range(nSamples):
#        Sigma_sample = samples_from_the_iw_prior[i]
#        plot_ellipse_using_cholesky(np.zeros(2),Sigma_sample,scale=3,lw=1)
        mu_sample,Sigma_sample = samples_from_the_niw_prior[i]
        plot_ellipse_using_cholesky(center=mu_sample,Sigma=Sigma_sample,scale=3,lw=1)    
    my_axes()
    
#    plt.title('draws$\sim$IW prior')
    plt.title('draws$\sim$NIW prior')
    iw_posterior_mode = gaussian_inference.iw_posterior.mode()
    
    
    
    nSamples=10
    
    

    m_posterior = gaussian_inference.m_posterior
    
    plt.subplot(234)
#    plt.subplot(231)
    plot_data(x,ms=3)
    plot_ellipse_using_cholesky(mu,Sigma=Sigma,scale=3,color='b',lw=1)
    plot_ellipse_using_cholesky(mu_mle,Sigma=Sigma_mle,scale=3,color='r',lw=1)
    plot_ellipse_using_cholesky(center=m_posterior,Sigma=iw_posterior_mode,scale=3,color='g',lw=1)
    plt.legend(['data','GT','MLE','MAP'],loc='lower left',fontsize=10)
    my_axes()
    
    plt.title('GT/MLE/MAP')
    
    
    
    plt.subplot(235)
#    for i in range(nSamples):
#        Sigma_sample = samples_from_the_iw_posterior[i]
#        plot_ellipse_using_cholesky(center=np.zeros(2),Sigma=Sigma_sample,scale=3,lw=1)    
#    my_axes()
#    plt.title('draws$\sim$IW posterior')
#    plt.subplot(236)
    plot_data(x,ms=3)
    for i in range(nSamples):
        mu_sample,Sigma_sample = samples_from_the_niw_posterior[i]
        plot_ellipse_using_cholesky(center=mu_sample,Sigma=Sigma_sample,scale=3,lw=1)    
    my_axes()
    plt.title('draws$\sim$NIW posterior')
    
    
    plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.91,wspace=0.05,hspace=0.3)
    

    return name,fig
if __name__ == '__main__':
    import os
    for N in (10,100,1000,10000):
        for nu in (2,10,100,1000):   
            kappa=nu # pseduocounts normal  
            name,fig=main(N=N,nu=nu,kappa=kappa)
            filename=name+'.jpg'
            filename = os.path.join('./figs_NIW',filename)
            
            print 'saving ',filename    
            plt.savefig(filename,dpi=100)
            
        