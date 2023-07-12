# The dynamics of integrated information in variational inference

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/bayes.jpg?raw=true" alt="Bayes and consciousness" width="600 px" />

Variational inference is a method from machine learning to approximate difficult-to-compute posterior densities. It is also used as a model for various cognitive processes in the context of the "Bayesian Brain" hypothesis. Integrated information is an information-theoretic measure that is thought to reflect the conscious level of a (biological) system. Research addressing the relationship between approximate Bayesian inference as a model for perception and integrated information as a proxy for conscious processing has so far been only conceptually formulated. Our goal was to 1) propose our idea of how to  link the two, and 2) provide a mathematical framework. To this end, we used an integrated information measure for autoregressive time-series models proposed by Adam Barrett and Anil Seth (2011) in black-box variational inference as established by Ranganath, Gerrish, and Blei (2016). Preliminary results show that in the course of approximate Bayesian inference, integrated information converges to zero. Overall, this gives a combined framework for formal models of consciousness (integrated information) and perception (variational inference).

## Black Box Variational Inference and Stochastic Gradient Descent

Black box variational inference can be demonstrated using the following visualization: A simpler model (the Gaussian distribution) is used to approximate a more complex posterior distribution (the funnel-like distribution) in an optimization framework (i.e., with each iteration, the approximating distribution comes closer to the true posterior distribution).

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/variation_inf.gif?raw=true" alt="Variational Inference" width="400 px" />

The following figure shows the approximating distribution and true posterior distribution for different iterations in the Black Box Variational Inference process.

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/2D_variational_target_distr.png?raw=true" alt="2D plot"/>

All code to generate the results and plots can be found in the two Python scripts. It simulates the trajectory of Stochastic Gradient Descent applied to the optimization of Black Box Variational Inference and subsequently calculates integrated information across time steps.

## Three dimensional BBVI with Stochastic Gradient Descent (SGD)

In the case of a three dimensional target distribution (that is to be approximated), one example result of the dynamics of information integration (as a proxy of consciousness) is shown below. In the top row, lighter color represents later iteration steps, i.e. the yellow-green cluster shows the convergence of the Black Box Variational Inference method.

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/3D_scatterplot_with_phi.png?raw=true" alt="3D plot"/>
