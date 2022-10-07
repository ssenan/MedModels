import torch

def beta_schedule(schedule_name, beta_start, beta_end, timestep, s=0.008):
    if schedule_name == 'linear':
        beta = torch.linspace(beta_start, beta_end, timestep)
    
    elif schedule_name == 'cosine':
        """
        See Phil Wang implementation
        """
        steps = timestep + 1
        x = torch.linspace(0, timestep, steps)
        alphas_cumprod = torch.cos(((x / timestep) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        beta =  torch.clip(beta, 0.0001, 0.9999)
        return beta

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule_name}")


class Diffusion:
    def __init__(self, beta, device):
        self.device = device
        self.beta = beta.to(self.device)
        alpha = 1 - beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def diffuse(self, x0, t):
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)

        x0 = torch.tensor(x0.to(self.device))
        eps = torch.rand_like(x0).to(self.device)
        alpha_hat_t = self.alpha_hat.index_select(0,t)

        xt = torch.sqrt(alpha_hat_t)*x0 + \
            torch.sqrt(1-alpha_hat_t)*eps

        return xt

    def denoise(self, xt, eps, t):
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)

        xt = torch.tensor(xt).to(self.device)
        alpha_hat_t = self.alpha_hat.index_select(0,t)

        x0_hat = 1/torch.sqrt(alpha_hat_t)*xt - \
            torch.sqrt(1-alpha_hat_t)/torch.sqrt(alpha_hat_t)*eps
        
        return x0_hat

    
    def reverse(self, xt, eps, t):
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
            
        beta_t = self.beta.index_select(0,t)
        alpha_t = self.alpha.index_select(0,t)
        alpha_hat_t = self.alpha_hat_t.index_select(0,t)
        alpha_hat_prev = self.alpha_hat_t.index_select(0,t-1)

        c1 = torch.sqrt(alpha_hat_prev)*beta_t / 1-alpha_hat_prev
        c2 = torch.sqrt(alpha_t)*(1-alpha_hat_prev) / 1-alpha_hat_prev

        x0_hat = self.denoise(xt, eps, t)
        x_prev = c1*x0_hat + c2*xt

        return x_prev