# Gibbs_Sampling_MC
Monte Carlo Gibbs Sampling algorithm applied to Gaussian Mixture Model

"Given a population of observations coming from different sources for which we don't know which subpopulation a data point belongs to, how can we estimate the origin of every point of the dataset? In other words, how can we cluster the observations so that we have an idea of the probability distrubutions of the sources they come from?"

By making some assumtions about the nature of those sources and by making use of the Gibbs Sampling (GS) algorithm we can solve those questions and get a family of solutions for which we can then see the most probable outcome, allowing the model to learn the subpopulations automatically. Since subpopulation assignment is not known, this constitutes a form of unsupervised learning. In this project we master the Gibbs Sampling concept by not only explaining the theory behind but also applying it to a real case. 

This project is made up of two files: In the Monte_Carlo_Gibbs_Sampling.pdf file you'll find everything you need to know about GS algorithm based on several reliable sources and specifically, how to apply it to a Gaussian Mixture Model (GMM). Finally, in the Gibbs_sampler_GMM.py file we put the theory in practice by bulding a pythonian Gibbs Sampler and using it to sample from the Posterior distribution of the GMM dataset (Iris Dataset).
