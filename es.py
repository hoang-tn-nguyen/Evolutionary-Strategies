import torch
import torch.nn as nn

# --- Core Modules ---
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
        ctx.mark_non_differentiable(theta)
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
    
# --- Helper Functions ---
def model_to_params(model, flatten=True):
    '''
    Get all parameters of the model
    Input:
        model: nn.Module: the input model
    Return:
        params: list: A list of parameter groups
    '''
    params = []
    for param in model.parameters():
        if flatten:
            params.append(param.view(-1))
        else:
            params.append(param)
    return params

def params_to_model(params, model):
    '''
    Import parameters to the model (inplace/overwriting)
    Input:
        params: list or torch.Tensor: A list of parameter groups or a 1-D tensor concatenating all parameter groups 
        model: nn.Module: the input model
    Return:
        model: nn.Module: the newly updated model
    '''
    if isinstance(params, list): # A list of 1-D Tensors
        for i, param in enumerate(model.parameters()):
            param[:] = params[i].view(*param.shape) # overwrite on the existing sequence
    elif isinstance(params, torch.Tensor):
        # Make sure that this is 1-D Tensor (i.e. a concatenation of multiple 1-D tensors)
        params = params.view(-1) 
        s = 0 # Start of sequence
        for param in model.parameters():
            e = s + param.numel() # End of sequence
            param[:] = params[s:e].view(*param.shape) # overwrite on the existing sequence
            s = e
    else:
        raise TypeError('params must be a list of 1-D torch.Tensor or 1-D torch.Tensor')
    return model

def normalize(input, eps=1e-9):
    return (input - input.mean()) / (input.std() + eps)