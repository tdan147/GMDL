#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import time
import numpy as np
from pylab import plt
from gmm_plt import *
from GaussianInference import GaussianInference
from CategoricalInference import CategoricalInference




class GMMInference(object):
    def __init__(self,data_dim,K,hyperparams,colors):
        self.data_dim=data_dim # data dimension
        self.K=K # number of components
        self.hyperparams=hyperparams
        # NIW:
        Psi=hyperparams['Psi']  
        m=hyperparams['m']
        kappa=hyperparams['kappa']
        nu=hyperparams['nu']
        
        if len(m)!=data_dim:
            raise ValueError(len(m),data_dim)
        if Psi.shape!=(data_dim,data_dim):
            raise ValueError( Psi.shape,'!=',(data_dim,data_dim)) 
        
        # Dirichlet:
        alphas=hyperparams['alphas']
        
        # Components
        self.comps = [GaussianInference(Psi=Psi,m=m,kappa=kappa,nu=nu) for k in range(K)]   
        # Weights
        self.categorical_inference = CategoricalInference(K=K,alphas=alphas)

        if colors.shape != (K,3):
            raise ValueError(colors.shape,'!=',(K,3))
        self.colors=colors
    def gibbs_sampling(self,x,nIters,N,tf_savefig):
        colors=self.colors
        data_dim=self.data_dim
        if x.shape != (data_dim,N):
            raise ValueError(x.shape,'!=',(data_dim,N))
        K=self.K
        z_est = self.z_est = np.random.choice(a=np.arange(K),size=N)   
       
        Ts = self.Ts = [None]*K # sufficent stats of for the Gaussians
        mus_est = self.mus_est = [None]*K # means 
        Sigmas_est = self.Sigmas_est = [None]*K # covs
       
        comps=self.comps
        categorical_inference = self.categorical_inference
        for step in range(nIters):
            print ('iter',step  ) 
            plt.figure(100)
            
            plt.clf()      
              
            if data_dim==2:
                plt.subplot(131) 
                for k in range(K):                    
                    plot_data(x[:,z_est==k],ms=5,color=colors[k])
            # sample Gaussian params conditioned on the labels
            for k in range(K):
                gaussian_inference=comps[k]

                Ts[k]=gaussian_inference.calc_suff_stats(x[:,z_est==k])
                T1,T2=Ts[k]
                N_k = (z_est==k).sum()
                gaussian_inference.calc_posterior_hyper_params(N=N_k,T1=T1,T2=T2)
                mu_new,Sigma_new=gaussian_inference.single_sample_from_the_niw_posterior()
            
                mus_est[k]=mu_new
                Sigmas_est[k]=Sigma_new  
                   
         
                color=colors[k]
                mu_est=mus_est[k]
                Sigma_est=Sigmas_est[k]          
    
    
                if data_dim==2:       
                    plot_ellipse_using_cholesky(mu_est,Sigma=Sigma_est,scale=3,color=color,lw=3)    
                    my_axes(b=24)        
                    plt.title('iter {}; $\mu_k,\Sigma_k$|$z,$data'.format(step))
                
            
            # sample pi conditioned on the labels
                
            Ns = categorical_inference.calc_suff_stats(z_est)  
            categorical_inference.calc_posterior_hyper_params(Ns)   
            pi_est = categorical_inference.single_sample_from_the_dirichlet_posterior()  
            
        
        
            plt.figure(100)        
            plt.subplot(132)         
            plt_pi(pi_est,colors)
            
            plt.title('iter {}; $\pi|z$,data'.format(step))
        
            print ('pi_est: ',pi_est)
           
            # sample labels conditioned on the Gaussian params and pi
            d = np.zeros((N,K))
            l = np.zeros((N,K))
            for k in range(K):
                y=x-mus_est[k][:,np.newaxis]
                Q=np.linalg.inv(Sigmas_est[k])
        #        d[:,k]=(y[0,:]*Q[0,0]+y[1,:]*Q[1,0])*y[0,:]+(y[0,:]*Q[0,1]+y[1,:]*Q[1,1])*y[1,:]
                
                for i in range(N):
                    d[i,k]=y[:,i].dot(Q).dot(y[:,i])
                l[:,k]=-0.5*d[:,k]        
                
                junk,logdet = np.linalg.slogdet(Sigmas_est[k])
                l[:,k]-= logdet
                l[:,k]+= np.log(pi_est[k])
            
                 
            M = l.max(axis=1)
            l-=M[:,np.newaxis]
            
            total = np.exp(l).sum(axis=1)
            probs = np.exp(l) / total[:,np.newaxis]
            
            
            # There is a faster way to do the sampling below
            # (without a loop) using the "inverse" CDFs.
            # I have it coded somewhere...
            for i in range(N):
                z_est[i]=np.random.choice(a=np.arange(K),p=probs[i])
            
            
         
            plt.figure(100)
            
            if data_dim==2:
                plt.subplot(133)
                
                for k in range(K):
                    plot_data(x[:,z_est==k],ms=5,color=colors[k])
                my_axes(b=24)
                plt.title('iter {}; $z$|$\mu_k,\Sigma_k,\pi,$data'.format(step))
        
            
            plt.figure(100)
            plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.91,wspace=0.05,hspace=0.5)
            
            if tf_savefig:
                name ='K_{0:02}_N_{1:05}_nu_{2:04}_alpha_{3:04}_iter_{4:03}'.format(K,N,nu,alpha,step)
                name = './figs_GMM_Gibbs/'+name+'.jpg'
                print ('saving ',name   )
                plt.savefig(name,dpi=100)




if __name__ == "__main__":
    
    p = np.array([0.3,0.2,0.4,0.1])
    p = p/p.sum()
    K=len(p)
    np.random.seed(0)
    colors = np.random.rand(K,3) 
    
    N = 10000
   
    z = np.random.choice(a=np.arange(K),size=N,p=p)
    #z=np.sort(z)
    
    data_dim = 2 # data dimension 
    
    # Ground Truth means
    mu1 = np.array([15-3,5])
    mu2 = np.array([-2-3,3])
    mu3 = np.array([0-3,-7])
    mu4 = np.array([-15-3,0])
    
    
    # Ground Truth covs
    A = 3*np.array([[1,-.5],[0,1]]).astype(np.float)
    Sigma1 = A.dot(np.eye(2)).dot(A.T)
    A = 2*np.array([[1,.25],[0,1]]).astype(np.float)
    Sigma2 = A.dot(np.eye(2)).dot(A.T)
    A = 2* np.array([[2,.1],[0,1]]).astype(np.float)
    Sigma3 = A.dot(np.eye(2)).dot(A.T)
    A = np.array([[1,0],[0,10]]).astype(np.float)
    Sigma4 = A.dot(np.eye(2)).dot(A.T)
    
    
    mus = [mu1,mu2,mu3,mu4]
    Sigmas = [Sigma1,Sigma2,Sigma3,Sigma4]  
    
    
    
    
    x = np.zeros((2,N))
    for k in range(K):
        N_k = (z==k).sum()
        mu_k = mus[k]
        Sigma_k = Sigmas[k]
        x[:,z==k]=np.random.multivariate_normal(mean=mu_k,cov=Sigma_k,size=N_k).T
    
    
    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    for k in range(K):
        plot_data(x[:,z==k],ms=5,color=colors[k])
    for k in range(K):
        mu=mus[k]
        Sigma=Sigmas[k]   
        plot_ellipse_using_cholesky(mu,Sigma=Sigma,scale=3,color=colors[k],lw=3)
        my_axes(b=24)
    plt.title('Ground Truth')


    
    hyperparams=dict()
    hyperparams['Psi']=np.eye(2)
    hyperparams['m'] = np.zeros(2)
    hyperparams['kappa']=2
    hyperparams['nu']=10
    hyperparams['alphas']= np.ones(K) * 100   
        
    
    gmm_inference = GMMInference(data_dim=data_dim,K=K,hyperparams=hyperparams,
                                 colors=colors)
    
    tf_savefig=False
    tic = time.clock()
    gmm_inference.gibbs_sampling(x=x,nIters=40,N=N,tf_savefig=tf_savefig)
    toc=time.clock()
    print ( 'time:',toc-tic    )

    



