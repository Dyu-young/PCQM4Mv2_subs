import torch
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration is no longer globally convergent.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized over Newton-schulz
    Muon internally uses Newton-Schulz iterations to orthogonalize the update steps.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adamw_params=None, adamw_lr=0.001, adamw_betas=(0.9, 0.95), adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas, adamw_wd=adamw_wd)
        
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.ndim < 2:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf
                    p.data.add_(update, alpha=-lr)
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Orthogonalize
                g_orth = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                scale = max(1, g.size(0)/g.size(1))**0.5
                p.data.add_(g_orth, alpha=-lr * scale)

        return loss
