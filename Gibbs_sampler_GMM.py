# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:18:24 2020

@author: Eric Bataller Thomas
"""
#!/usr/bin/env python

"""
Posterior sampling for Gaussian Mixture Model using Gibbs sampler
"""

# Numerical Computing Libraries

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn import datasets

class MixGaussGibbsSampler():
      def __init__(self, X, Z, m0=None , V0=None, nu0=None, S0=None,  alpha=None ,burn_in=50):
            """ This function initializes the Gibbs sampler with some standard parameters, 
            for how the Gibbs Sampler works
            burn_in = The number of iterations that we reach before assumed convergence 
                      to the stationary distribution.
            X = The random variables.
            Y = The labels for the variables --> a vector with random assignations should be introduced
            """
            # Add the variables to the class
            self.X = X # The data points
            self.Z = Z # The cluster assignments, this should be generated randomly if no info is available 
            self.burn_in = burn_in
            self.num_mixtures = self.Z.max() + 1
            self.num_points = X.shape[0]
            self.dim = X.shape[1]
            
            #init mu,sigmas,
            self.pis = np.zeros((self.num_mixtures, 1))
            self.mus = np.zeros((self.num_mixtures, X.shape[1]))
            self.sigmas = np.zeros((self.num_mixtures, self.dim, self.dim))
            self.lambdas = np.zeros((self.num_mixtures, self.dim, self.dim)) #The inverse of the sigmas
            self.nk = np.zeros((self.num_mixtures,1),dtype=int) #count of points of every cluster
            
            diagsig = np.zeros((self.num_mixtures, self.dim, self.dim)) #Diagonal scatter matix .We will need it to initialize S0 
            for k in range(0,self.num_mixtures):

                  # Gather all data points assigned to cluster k
                  assigned_indices = (self.Z == k)
                  nk = assigned_indices.sum()
                  self.nk[k]=nk
                  
                  # from x of class_k create the x_k_bar
                  class_k = self.X[assigned_indices, :]
                  x_bar = np.sum(class_k, axis=0)
                  x_bar = np.array(x_bar / float(nk))
                  
                  # first mu,sigma guess:
                  sig_bar = np.cov(class_k.T ,bias=True)
                  sig_diagonal = np.diag(np.diag(sig_bar))
                  self.mus[k,:] = x_bar
                  self.sigmas[k,:,:] = sig_bar
                  self.lambdas[k,:,:] = np.linalg.inv(sig_bar)
                  diagsig[k,:,:] = sig_diagonal
                   
      
            #Priors 
            if not m0: #if I have no prior info about the means of mus
                  self.m0 = self.mus 
            else:
                  self.m0 = m0
                  
            if not V0: #if I have no prior info about the variance of mu
                  I = np.identity(self.dim)
                  listI = []
                  for k in range(0,self.num_mixtures):
                        listI.append(I)
                  self.V0 = np.array(listI)*10000000 #uninformative mu prior --> V0 = inf*I
                  self.invV0 = np.array(listI)*(1/10000000)       
            else:
                  self.V0 = V0
                  self.invV0 = np.zeros((self.num_mixtures, self.dim, self.dim))
                  for k in range(0,self.num_mixtures):
                        self.invV0[k,:,:] = np.linalg.inv(self.V0[k,:,:])
                  
            if not S0: #if I have no info about the mean of the sigma prior 
                  self.S0 = diagsig #common choice
            else:
                  self.S0 = S0
                  
            if not nu0: #if I have no info about the strength of the sigma prior 
                  self.nu0 = self.dim + 2 #common choice
            else:
                  self.nu0 = nu0
            
            if not alpha: #if I have no info about the Dirichlet prior 
                  self.alpha = (100*np.ones(self.num_mixtures)/self.num_mixtures) #we set a concave distribution in the simplex with max probability around the center --> more or less same amount of points for each gaussian
            else:
                  self.alpha = alpha #it has to be a 1D array of shape (num_mixtures,)
            
            
            self.pis = st.dirichlet.rvs(self.alpha, 1).T #init pi vector --> all gaussians are equally probable
            self.iter_prob = [] # A list variable holding the total probability
                                # at each stage of the iteration
            pass

      def perform_gibbs_sampling(self, iterations=False):
            """ This function controls the overall function of the gibbs sampler, and runs the gibbs
            sampling routine.
            iterations = The number of iterations to run, if not given will run the amount of time 
                         specified in burn_in parameter
            """
            if not iterations:
                num_iters = self.burn_in
            else:
                num_iters = iterations
                
            # Plot the initial set up
            self.plot_points("Initial Random Assignments")
            
            # Run for the given number of iterations
            for i in range(num_iters):
                self.sample_mu()
                self.sample_sigma()
                self.sample_pi()
                self.sample_z()
                print(i+1,"/",num_iters)
                
            # Plot the final mixture assignments
            self.plot_points("Final Mixture Assignments")
            
            return self.Z, self.mus, self.sigmas, self.pis

      def sample_mu(self):
            """ This function samples mu from N(mu| mk,Vk) and updates them"""
            
            for k in range(0,self.num_mixtures): 
                
                # Gather all data points assigned to cluster k
                assigned_indices = (self.Z == k)
                class_x = self.X[assigned_indices, :]
                x_bar = np.sum(class_x, axis=0)
                x_bar = np.array(x_bar / float(self.nk[k]))
                
            
                # Covariance of posterior Vk
                invVk = np.zeros((self.dim, self.dim))
                invVk = self.invV0[k,:,:] + self.nk[k]*self.lambdas[k,:,:]
                cov_postVk = np.linalg.inv(invVk)
                
                # Mean of posterior mk
                left = cov_postVk
                right = self.invV0[k,:,:] @ self.m0[k] + self.nk[k]*self.lambdas[k,:,:] @ x_bar
                mu_postmk = left @ right
                
                # Draw new mean sample from posterior
                self.mus[k,:] = st.multivariate_normal.rvs(mu_postmk, cov_postVk)
                
            pass
      
      def sample_sigma(self):
            """ This function samples sigmas from IW(sigma| Sk, nuk) and updates them"""

            for k in range(0,self.num_mixtures): 
                
                # Gather all data points assigned to cluster k
                assigned_indices = (self.Z == k)
                class_x = self.X[assigned_indices, :]
            
                # Posterior scatter matrix Sk
                a = np.array(self.mus[k,:])[np.newaxis]
                dist = class_x - np.tile(a, (int(self.nk[k]),1))
                Sk = self.S0[k,:,:] + (dist.T @ dist)
                
                # Posterior strenght nuk
                nuk = self.nu0 + self.nk[k]

                # Draw new mean sample from posterior
                self.sigmas[k,:,:] = st.invwishart.rvs(int(nuk), Sk)
                self.lambdas[k,:,:] = np.linalg.inv(self.sigmas[k,:,:])     
                
            pass

      def sample_z(self):
            """ Now we will sample the cluster assignments given the mixture locations and sigmas"""
            for i in range(0,self.num_points):
                # Sample the cluster assignment
                """ This for performs one sampling assignment"""
                probs = np.zeros([self.num_points, self.num_mixtures])
                for k in range(0, self.num_mixtures):
                    p = self.pis[k]*st.multivariate_normal.pdf(self.X, mean=self.mus[k,:], cov=self.sigmas[k,:,:])
                    probs[:, k] = p
      
                # Normalize
                probs /= np.sum(probs, axis=1)[:, np.newaxis]
                
                # For each data point, draw the cluster assignment
                for i in range(0,self.num_points):
                    z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
                    self.Z[i] = z
                
                # New count of cluster points  
                for k in range(0,self.num_mixtures):
                      assigned_indices = (self.Z == k)
                      nk = assigned_indices.sum()
                      self.nk[k]=nk  
            pass
      

      def sample_pi(self):
            a = self.alpha + self.nk.T
            self.pis = st.dirichlet.rvs(a[0,:], 1).T
            
      
      def plot_points(self, title):
            """ This plots the points and the mus in a scatter plot"""
            fig, ax = plt.subplots()
            datasets = []
            for k in range(self.num_mixtures):
                # Assigned indices
                assigned_indices = (self.Z == k)
                datasets.append(self.X[assigned_indices, :])
            
            # Now let's put the scatter plots onto the scene.
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for j, data in enumerate(datasets):
                ax.scatter(data[:, 2], data[:, 3], color=colours[j], cmap=plt.cm.Set1, edgecolor='k')
                
            
            ax.set_title(title)
            plt.grid(True)
            plt.show()
            pass
#ENDCLASS
'''
Function to generate a test dataset:
      
def generate_test_data():
      """ This is the a test script to generate test data"""
      # All of these will be assumed 2-D data
      # Let's generate a mixture of three gaussians
      u_0 = np.array([-2.0,-2.0])
      u_1 = np.array([0.0,2.0])
      u_2 = np.array([2.0,-2.0])
      u= [u_0,u_1,u_2]
      # sigmas for each of the data
      sigmas = [1, 1, 1]
      points_per_cluster = 50
      
      # Now initialize the data variables
      data0 = np.random.randn(points_per_cluster,2)*sigmas[0] + u_0
      data1 = np.random.randn(points_per_cluster,2)*sigmas[1] + u_1
      data2 = np.random.randn(points_per_cluster,2)*sigmas[2] + u_2
      X = np.vstack((data0, data1, data2))
      print(X.shape)
      # Now random init the cluster assignments
      rand_Y = np.random.randint(0, 3, points_per_cluster*3)
      return X, rand_Y, u
'''

if __name__ == "__main__":
      # Test script for checking my module
      
      #X, Z,u = generate_test_data() #Z has 150 elements = elements in iris dataset
      
      #For the purposes of this exercise, we will be using the Iris dataset.
      #We can easily obtain it by using the load_iris function provided by sklearn:
      iris = datasets.load_iris()
      Y = iris.data
      Z = np.random.randint(0, 3, 150)
      
      # Now initalize the gibbs sampler
      gs = MixGaussGibbsSampler(Y, Z)
      gs.perform_gibbs_sampling()

      print(gs.Z)
      print(gs.mus)
      
      #making Y and Z a dataframe so i can plot it with seaborn.
      d = {'sepal length': Y[:,0], 'sepal width': Y[:,1], 'petal length': Y[:,2],'petal width': Y[:,3], 'assig': gs.Z}
      df = pd.DataFrame(data=d)
      sns.set_style("whitegrid");
      g = sns.pairplot(df, hue='assig',palette="husl")



