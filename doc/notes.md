## Profiling / optimization notes
Dumped a profiler trace of c5a81deb0aaa0ae1cbcff85e3b18e7cae9e07c94 to /datadrive/build/chrome_trace (load into Chrome profiling tools)
  - Looks like a very large percentage of the time is spent in TakeBackward

Also, on master there are JIT optimizations which are probably useful, given how many operations we have in the forward pass: https://gist.github.com/zdevito/4aafb8fea540dfd7644cc09b2ce78230

## Posterior collapse


## Similar papers / lit review
- Less computationally intensive algorithms for filtering / smoothing on linear state space models:
  - Forward-backward
  - Kalman filter
  - RTS smoother

- Essentially, it's important to note the significant previous work on applying particle methods to non-Markovian models
  - Frigola et al., Identification of Gaussian Process State-Space Models with Particle Stochastic Approximation EM, 2014
  - Svensson et al., Identification of jump Markov linear models using particle filters, 2014
  - Wood et al., A new approach to probabilistic programming inference, 2014
  - Naesseth et al., Sequential Monte Carlo for Graphical Models, 2014
  - Lindsten et al., Particle Gibbs with Ancestor Sampling, 2014
  - Lindsten, F., & Schön, T. B. (2013). Backward Simulation Methods for Monte Carlo Statistical Inference. Foundations and Trends in Machine Learning, 6(1), 1–143.
  - Neural Adaptive Sequential Monte Carlo (Gu et al.)

- State-space LSTM Models with Particle MCMC inference
    - https://openreview.net/pdf?id=r1drp-WCZ (most recent)
    - https://arxiv.org/abs/1711.11179
    - ICLR comments are useful as well, it was rejected (https://openreview.net/forum?id=r1drp-WCZ)

## TODO
1. Extract single word sentiment labels from this torchtext garbage; fix evaluate bug in discrete pfilter; fix resample bug in discrete_pfilter
  - evaluate bug was 0-temperature not being handled correctly in RelaxedCategorical
  - resample bug is serious beeswax, took forever to debug - it was torch.multinomial being crap, I think? reprod, but there was a fix a few days ago, so recompiling latest master.


## random notes
note: random init for larger_hmm_data.pt with z_dim=25, x_dim=50 was -14695.8707123 
