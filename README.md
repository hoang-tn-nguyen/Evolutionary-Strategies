# Evolutionary-Strategies
An implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"

This implementation supports
- PyTorch-GPU
- Compute gradients w.r.t. mean and standard deviation (the original paper used a constant standard deviation)
- Import/Export parameters to working models

Note:
- ES is very sensitive to the initial mean, standard deviation, and learning rate. Hyper-parameter tuning is very important.
