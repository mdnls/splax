# Splats + Jax = SPLAX

**Mara, Jessie**

We run a battery of comparisons between splat regression models, Kolmogorov Arnold Networks, and Multilayer Perceptrons. We use a series of optimizations to scale up training, including: 
- SVD/QR covariance parametrization (accelerate gradient computations by using an easily invertible matrix decomposition)
- Sparsity aware backprop (for each input, only a sparse subset of neurons are active when computing the output)

We also implement the following regularization schemes: 
- Entropy penalization/stochastic optimization
- Fisher-Rao exploration (particle deletion, teleportation)
- Sparse particle revulsion (multipole method?) 
- 