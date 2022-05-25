#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import numpy as np
from scipy.stats import invwishart


class GaussianInference(object):
    def __init__(self,Psi,m,kappa,nu):
        self.n=len(m)
        self.Psi=Psi
        self.m=m
        self.kappa=kappa
        self.nu=nu
        self.data_dim = data_dim = len(m)
        if Psi.shape != (data_dim,data_dim):
            raise ValueError(Psi.shape ,'!=', (data_dim,data_dim))

        self.iw_prior = invwishart(df=nu,scale=Psi*nu)
#    iw_prior_mean = iw_prior.mean()
#    iw_prior_mode = iw_prior.mode()

    def calc_posterior_hyper_params(self,N,T1,T2):
        nu=self.nu
        kappa=self.kappa
        m=self.m
        Psi=self.Psi
        
        nu_posterior = nu+N
        kappa_posterior = kappa + N
        if m.shape != (self.n,):
            raise ValueError
        if Psi.shape != (self.n,self.n):
            raise ValueError    
        
        data_dim = self.data_dim
        if len(T1)!= data_dim:
            raise ValueError(len(T1),'!=',data_dim)
        
        
        m_posterior = 1.0/kappa_posterior* (kappa*m + T1)
        Psi_posterior = 1.0/nu_posterior*(nu * Psi + kappa*np.outer(m,m)  \
                         -kappa_posterior*np.outer(m_posterior,m_posterior)+  T2) 
    
        if m_posterior.shape != (self.n,):
            raise ValueError
        if Psi_posterior.shape != (self.n,self.n):
            raise ValueError
        self.kappa_posterior=kappa_posterior
        self.nu_posterior=nu_posterior
        self.Psi_posterior=Psi_posterior
        self.m_posterior=m_posterior
                                    
        n=self.n
        if Psi_posterior.shape != (n,n):
            raise ValueError(Psi_posterior.shape,n)
        self.iw_posterior = invwishart(df=nu_posterior,scale=Psi_posterior*nu_posterior)

    def sample_from_the_iw_prior(self,nSamples):
        return self.iw_prior.rvs(size=nSamples)

    def sample_from_the_iw_posterior(self,nSamples):
        n=self.n
        result = self.iw_posterior.rvs(size=nSamples)
        
        if result.shape != (nSamples,n,n):
            if nSamples==1:
                raise NotImplementedError
            raise ValueError(result.shape,nSamples)
        return result
    def single_sample_from_the_iw_posterior(self):        
        return self.iw_posterior.rvs(size=1)    
        
    def sample_from_the_niw_prior(self,nSamples):
        n=self.n
        kappa=self.kappa
        m=self.m
        samples_from_the_iw_prior = self.sample_from_the_iw_prior(nSamples)
        
        samples_from_the_niw_prior = []           
        for i in range(nSamples):        
            Sigma_sample = samples_from_the_iw_prior[i]
            if Sigma_sample.shape != (n,n):
                raise ValueError(Sigma_sample.shape)
            try:
                mu_sample = np.random.multivariate_normal(mean=m,cov=
                                                          1./kappa*Sigma_sample)
            except:
                raise ValueError(Sigma_sample)
                
            samples_from_the_niw_prior.append([mu_sample,Sigma_sample])
        return samples_from_the_niw_prior
    def sample_from_the_niw_posterior(self,nSamples):
        n=self.n
        kappa_posterior=self.kappa_posterior
        m_posterior=self.m_posterior
        samples_from_the_iw_posterior = self.sample_from_the_iw_posterior(nSamples)
        
        samples_from_the_niw_posterior = []           
        for i in range(nSamples):        
            Sigma_sample = samples_from_the_iw_posterior[i]
            if Sigma_sample.shape != (n,n):
                raise ValueError(Sigma_sample.shape)
            try:
                mu_sample = np.random.multivariate_normal(mean=m_posterior,cov=
                                                          1./kappa_posterior*Sigma_sample)
            except:
                raise ValueError(Sigma_sample)
                
            samples_from_the_niw_posterior.append([mu_sample,Sigma_sample])
        return samples_from_the_niw_posterior
    
    def single_sample_from_the_niw_posterior(self):
        n=self.n
        kappa_posterior=self.kappa_posterior
        m_posterior=self.m_posterior               
        Sigma_sample = self.single_sample_from_the_iw_posterior()
        mu_sample = np.random.multivariate_normal(mean=m_posterior,cov=
                                                      1./kappa_posterior*Sigma_sample)        
        return [mu_sample,Sigma_sample]

     
    def calc_suff_stats(self,x):
        T1 = x.sum(axis=1)  # T1 is a vector of length d        
        T2 = x.dot(x.T) # T2 is dxd
        if len(T1)!=self.data_dim:
            raise ValueError(len(T1),'!=',self.data_dim)
        return T1,T2 
    @staticmethod
    def calc_mle(T1,T2,N):
        mu = T1/N
        Sigma = T2/N - np.outer(mu,mu)
        return mu,Sigma