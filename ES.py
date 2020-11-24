import torch
import torch.nn as nn

class Expectation(nn.Module):
    def __init__(self):
        '''
        Input:
            F: Values of F(x)
            P: Distribution P(x)
        Return:
            E[F(x)] = \sum{F(x).P(x)}
            => E[F] = \sum{F.P}
        '''
        super().__init__()
    
    def forward(self, F, P):
        return (F*P).sum()

class NormalAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, size, random_state):
        eps = torch.randn(size, *mu.shape, device=mu.device, generator=random_state)
        ctx.save_for_backward(eps, sigma)
        theta = mu + sigma * eps 
        ratio = torch.ones(len(eps), dtype=torch.float32, device=eps.device) / len(eps)
        return theta, ratio

    @staticmethod
    def backward(ctx, grad_theta, grad_ratio):
        eps, sigma = ctx.saved_tensors
        grad_mu, grad_sigma = None, None
        if ctx.needs_input_grad[0]:
            grad_mu = grad_ratio @ eps / sigma
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_ratio @ (eps**2) / sigma
        return (grad_mu, grad_sigma, None, None)

class Normal(nn.Module):
    def __init__(self, device='cpu', seed=0):
        super().__init__()
        self.device = device
        self.random_state = torch.Generator(device=self.device).manual_seed(seed)

    def forward(self, mu, sigma, size):
        '''
        Input:
            mu: torch.tensor: Mean of the distribution
            sigma: torch.tensor: Standard deviation of the distribution
            size: int: Number of samples to be drawn from the distribution
        Return:
            theta: torch.tensor: Samples drawn from the distribution (N,*mu.shape)
            ratio: torch.tensor: 1 / size (N)
        '''
        theta, ratio = NormalAutograd.apply(mu, sigma, size, self.random_state)
        return theta, ratio
    
if __name__ == "__main__":
    from tqdm import tqdm
    solution = torch.tensor([0.5, 1, -0.9], device='cpu')
    def f(w):
        return ((w - solution)**2).sum(dim=-1)

    npop = 500     # population size
    mu = torch.randn(3, requires_grad=True, device='cpu')
    sigma = torch.randn(3, requires_grad=True, device='cpu')
    p = Normal(mu.device)
    expectation = Expectation()
    optimizer = torch.optim.Adam([mu, sigma])
    
    best_mean = 1e9
    best_mu = mu
    prog_bar = tqdm(range(10000))
    for i in prog_bar: 
        samples, ratio = p(mu, sigma, npop) # (N,3)
        fitness = f(samples)
        scaled_fitness = (fitness - fitness.mean()) / (fitness.std())
        mean = expectation(scaled_fitness, ratio)
        
        optimizer.zero_grad()
        mean.backward()
        optimizer.step()
        
        if mean < best_mean:
            best_mean = mean
            best_mu = mu
            
        prog_bar.set_description("step: {}, mean fitness: {:0.5}, std fitness: {:0.5}".format(i, float(fitness.mean()), float(fitness.std())))
    print(best_mu)