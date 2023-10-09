# The dynamics of integrated information in variational inference

### The purpose of this repository 
This repository showcases some plots including the plotting code from a project on integrated information in variational inference. The codebase is not complete, i.e., using this repository, results can not be reproduced, and the readme and all other material shown here are not optimized for understandability. Its main purpose is merely to **showcase several plots**, **including the plotting code**, from this project. The repository may be improved in the future in terms of understandability and reproducibility.

## Introduction to the project 
Many problems in different domains can be cast as the approximation of complicated probability densities. *Variational inference* (VI, also known as approximate Bayesian inference) is a method from machine learning used for approximating such difficult-to-compute probability densities (Jordan et al., 1999). VI approaches have been extensively explored in neuroscience to investigate the brain’s capacity to operate in situations of uncertainty in a Bayes’ optimal way (i.e., close to what Bayesian models predict) (Clark, 2013). In this context, it has also been hypothesized that the brain is facing hard-to-calculate posterior densities equally by exploiting approximate solutions such as VI. 

*Integrated information* (normally symbolized by φ) denotes the idea that a system of interconnected elements, considered in as a “whole”, can encode information that goes beyond the information encoded by the sum of individual “parts” (Barrett and Seth, 2011). This idea has been operationalized in mathematically different ways (Tegmark, 2016), and explored in different scientific contexts - particularly so in consciousness science, where an entire theoretical framework called Integrated Information Theory (IIT) has been developed (Oizumi et al., 2014), pursuing the core hypothesis that consciousness arises as a result of high information integration. Both VI and integrated information have been influential and explored considerably – albeit separately from each other – in neuroscience. Thus, a link between the two so far is missing, begging the question of whether VI as an inference method and φ as a measure of dependency between a system’s parts are related – i.e., will systems performing optimal Bayesian inference display integrated information? Here, we used an integrated information measure for autoregressive time-series models proposed by Adam Barrett and Anil Seth (2011) in *black-box variational inference* (BBVI) with *stochastic gradient descent* (SGD) as established by Ranganath, Gerrish, and Blei (2016).    

A detailed overview in terms of Marr's three levels can be seen in the following graphic.

<iframe src="https://docs.google.com/viewer?url=https://github.com/nadinespy/BBVI_SGD/raw/main/Marr_levels_applied.pdf&embedded=true" width="800" height="600" frameborder="0"></iframe>


[Link to PDF](https://github.com/nadinespy/BBVI_SGD/raw/main/Marr_levels_applied.pdf)


## Black box variational inference with stochastic gradient descent in a 2D example

BBVI can be demonstrated using the following visualization: A simpler model (the Gaussian distribution) is used to approximate a more complex posterior distribution (the funnel-like distribution) in an optimization framework (i.e., with each iteration, the approximating distribution comes closer to the true posterior distribution). The full target distribution is approximated by estimating its mean and log variance parameters.

We use a callback function (in [BBVI_SGD_PHI.py](https://github.com/nadinespy/BBVI_SGD/blob/main/BBVI_SGD_PHI.py)) to create Matplotlib figures of the true and approximative distribution in real-time.


<style>
  .centered {
    text-align: center;
  }
</style>

<div class="centered">
  <img src="https://github.com/nadinespy/BBVI_SGD/blob/main/variation_inf.gif?raw=true" alt="Variational Inference" width="45%" />
</div>


The following figure shows the same BBVI process in single plots where approximative and true posterior distribution are shown for different iterations.

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/2D_variational_target_distr.png?raw=true" alt="2D plot"/>


## Integrated information in BBVI with SGD in a 3D example

One example result of the dynamics of information integration is shown below where we use a 3D target distribution (that is to be approximated using several mean and log variance parameters). In the upper plot from below, I use a 3D matplotlib plot (in [BBVI_SGD_3DPlottingWithPhi.py](https://github.com/nadinespy/BBVI_SGD/blob/main/BBVI_SGD_3DPlottingWithPhi.py)) to visualize how parameters converge during the approximation: axes depict target variables, and lighter color represents later iteration steps, i.e. the yellow-green cluster shows the convergence of the BBVI/SGD method.

In the lower plot, we see the trajectories of all three parameters (denoted as "MAPs" - maximum a posteriori - on the left y-axis) plotted over iteration steps on the x-axis. Integrated information (denoted as φ on the right y-axis) is calculated for each pair of adjacent time steps, and its time-series is plotted together with the other time-series.


<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/3D_scatterplot_with_phi.png?raw=true" alt="3D plot"/>


### References

Barrett, A. B. and Seth, A. K. (2011). Practical measures of integrated information for time-series data. PLoS computational biology, 7(1):e1001052.

Clark, A. (2013). Whatever next? predictive brains, situated agents, and the future of cognitive science. Behavioral and brain sciences, 36(3):181–204.

Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., and Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37(2):183–233.

Oizumi, M., Albantakis, L., and Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: integrated information theory 3.0. PLoS computational biology, 10(5):e1003588.

Ranganath, R., Gerrish, S., & Blei, D. M. (2013). Black box variational inference. arXiv preprint arXiv:1401.0118.

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/bayes.jpg?raw=true" alt="Bayes and consciousness" width="600 px" />

<img src="https://github.com/nadinespy/BBVI_SGD/blob/main/variation_inf.gif?raw=true" alt="Variational Inference"    width="400 px" />