import math
import torch
from torch.optim.optimizer import Optimizer, required
import time

# Coded based on the official pytorch implementation

class PU_Adam_optimizer(Optimizer):
    """Implement Adam optimizer for weightwise partial updating.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.001)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, eta=1., weight_decay=0.):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, eta=eta, weight_decay=weight_decay)
        super(PU_Adam_optimizer, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(PU_Adam_optimizer, self).__setstate__(state)
    
    @torch.no_grad()       
    def step(self, DPU_layers, first_step=False, closure=None):
        """Perform a single optimization step. Note that current version only supports a single parameter group. 

        Args:
            DPU_layers (iterable): iterable of DPU_module objects. 
            first_step (bool, optional): indicate if conducting the first step in DPU (default: False)
            closure (callable, optional): a closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']
            for i, (p_dpu, p) in enumerate(zip(DPU_layers, group['params'])):
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:                    
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['eta'] = group['eta']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                eta = state['eta']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if weight_decay != 0:
                    grad.add_(weight_decay, p)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                numer = (lr * eta / bias_correction1) * exp_avg

                # Create sparse updating according to the mask 
                incr_step = torch.mul(-torch.div(numer, denom), p_dpu.mask)

                # If conducting the first step, update the local contribution
                if first_step:
                    p_dpu.local_contribution.add_(-torch.mul(incr_step, grad))
                    
                p.add_(incr_step)  
                
        return loss


class PU_SGD_optimizer(Optimizer):
    """Implement stochastic gradient descent optimizer (optionally with momentum) for weightwise partial updating.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.1)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        
    """
    def __init__(self, params, lr=0.1, momentum=0., weight_decay=0., nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov momentum requires a positive momentum")
        super(PU_SGD_optimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PU_SGD_optimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, DPU_layers, first_step=False, closure=None):
        """Perform a single optimization step. Note that current version only supports a single parameter group. 

        Args:
            DPU_layers (iterable): iterable of DPU_module objects. 
            first_step (bool, optional): indicate if conducting the first step in DPU (default: False)
            closure (callable, optional): a closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            lr = group['lr']
            for i, (p_dpu, p) in enumerate(zip(DPU_layers, group['params'])):
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)
                
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    if nesterov:
                        incr_step = -lr*(grad.add(buf, alpha=momentum)) 
                    else:
                        incr_step = -lr*buf 
                else:
                    incr_step = -lr*grad
                
                # Create sparse updating according to the mask    
                incr_step.mul_(p_dpu.mask)      
                
                # If conducting the first step, update the local contribution    
                if first_step:
                    p_dpu.local_contribution.add_(-torch.mul(incr_step,grad))
                    
                p.add_(incr_step)  
                
        return loss


