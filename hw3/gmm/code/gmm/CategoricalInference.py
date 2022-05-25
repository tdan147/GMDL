#! /usr/bin/env python
"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import numpy as np
from scipy.stats import dirichlet

class CategoricalInference(object):
    def __init__(self,K,alphas):
        self.K = K
        self.alphas=alphas
        self.dirichlet_prior = dirichlet(alpha=alphas)
    def calc_posterior_hyper_params(self,Ns):
        if Ns.shape != self.alphas.shape:
            raise ValueError(Ns.shape,self.alphas.shape)
        self.alphas_posteior = self.alphas + Ns
        self.dirichlet_posterior = dirichlet(alpha=self.alphas_posteior)
    
    def sample_from_the_dirichlet_prior(self,nSamples):
        if nSamples==1:
            raise NotImplementedError
        return self.dirichlet_prior.rvs(size=nSamples)
    def sample_from_the_dirichlet_posterior(self,nSamples):
        if nSamples==1:
            raise NotImplementedError
        return self.dirichlet_posterior.rvs(size=nSamples)
    def single_sample_from_the_dirichlet_posterior(self):
        return self.dirichlet_posterior.rvs(size=1)[0]
    
    
    def calc_suff_stats(self,z):
        return np.bincount(z,minlength=self.K)