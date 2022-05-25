#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""

import numpy as np
from pylab import plt
from CategoricalInference import CategoricalInference
from gmm_plt import plt_pi,plt_normalized_counts,plt_counts

def main(N= 100,alpha=100,K=5):
    name='K_{0:03}_N_{1:06}_alpha_{2:06}'.format(K,N,alpha)
    
    #  create the Ground Truth distribution. 
    p=np.ones(K)/K
    p[0]*=4
    if K>=3:
        p[2]*=2
        p[2]*=0.2
    p/=p.sum()
    
    np.random.seed(0)
    colors = np.random.rand(K,3)
    
    # create count data
    
    z = np.random.choice(a=np.arange(K),size=N,p=p)
           
    
    alphas = np.ones(K) * alpha
    categorical_inferece = CategoricalInference(K=K,alphas=alphas)
    
    
    Ns = categorical_inferece.calc_suff_stats(z)  
    
    categorical_inferece.calc_posterior_hyper_params(Ns) 
    
    
    nSamples = 8
    samples_from_the_dirichlet_prior = categorical_inferece.sample_from_the_dirichlet_prior(nSamples)
    
    
    samples_from_the_dirichlet_posterior = categorical_inferece.sample_from_the_dirichlet_posterior(nSamples)
    
    
    
    fig=plt.figure(1)
    plt.clf()
    
    
    for i, pi in enumerate(samples_from_the_dirichlet_prior):
        plt.subplot(3,nSamples,i+1)
        plt_pi(pi,colors)
        plt.title(r'$\pi \sim$ Dir$(\vec{\alpha})$')
        
        
    for i, pi in enumerate(samples_from_the_dirichlet_posterior):
        plt.subplot(3,nSamples,i+1+2*nSamples)
        plt_pi(pi,colors)
        plt.title(r'$\pi \sim$ Dir$(\vec{\alpha}^*)$')

    
    plt.subplot(3,nSamples,1*nSamples+1)
    plt_pi(p,colors)
    plt.title('GT $\pi$')
    plt.subplot(3,nSamples,1*nSamples+2)
   
    plt.ylim(0,N)
    plt_counts(Ns,colors)
    plt.title('counts; $N={0}$'.format(N))
    plt.subplot(3,nSamples,1*nSamples+3)
    plt_normalized_counts(Ns,colors)
    
    plt.title('noramalized')
    
    
    plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.91,wspace=0.5,hspace=0.4)
        
    return name,fig
if __name__ == "__main__":
    for K in (3,5,10):
        for alpha in (1,10,100,100):
            for N in (10,100,1000):                             
                name,fig = main(N= N,alpha=alpha,K=K)  
    
#                filename=name+'.png'
                filename=name+'.jpg'
                filename = os.path.join('./figs_catdir',filename)
                print 'saving ',filename    
                plt.savefig(filename,dpi=200)
    
    
    
    
    
    
    



