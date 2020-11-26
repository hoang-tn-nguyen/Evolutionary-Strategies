# Evolutionary-Strategies
An implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"

## This implementation supports
- PyTorch-GPU
- Compute gradients w.r.t. mean and standard deviation (the original paper used a constant standard deviation)
- Import/Export parameters to working models

*Note: ES is very sensitive to hyper-parameters such as population size, mean, standard deviation, and learning rate.*

## To do
Implementing Novelty Search "Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents"

## References
[1] Conti, E., Madhavan, V., Such, F. P., Lehman, J., Stanley, K., & Clune, J. (2018). Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. In Advances in neural information processing systems (pp. 5027-5038).\
[2] Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864.\
