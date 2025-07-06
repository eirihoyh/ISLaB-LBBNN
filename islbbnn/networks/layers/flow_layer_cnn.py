import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.flows import PropagateFlow

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'
# Z_FLOW_TYPE = 'RNVP'
# R_FLOW_TYPE = 'RNVP'

class BayesianConv2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size,
            num_transforms, 
            padding=0, 
            stride=1,
            lower_init_lambda=-10, 
            upper_init_lambda=15, 
            a_prior=0.1, 
            std_prior=1.):
        super().__init__()

        self.stride = stride
        self.padding = padding

        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size)

        elif type(kernel_size) == tuple:
            kernel = kernel_size

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]).uniform_(-1.2, 1.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_channels, in_channels, kernel[0], kernel[1]))
        self.weight_sigma = torch.empty(self.weight_rho.shape)

        # weight priors
        self.mu_prior = torch.zeros((out_channels, in_channels, kernel[0], kernel[1]), device=DEVICE)
        self.sigma_prior = (self.mu_prior + std_prior).to(DEVICE)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]).uniform_(lower_init_lambda, upper_init_lambda))
        self.alpha_q = torch.empty(self.lambdal.shape)

        # model prior
        self.alpha_prior = (self.mu_prior + a_prior).to(DEVICE)

        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(-9 + 1. * torch.randn(out_channels))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias prior
        self.bias_mu_prior = torch.zeros(out_channels, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + std_prior).to(DEVICE)
        
        self.q0_mean = nn.Parameter(1 * torch.randn(out_channels))
        self.q0_log_var = nn.Parameter(-9 + 1 * torch.randn(out_channels))
        # auxiliary variables c, b1, b2 defined in eqs. (11), (12)
        self.r0_c = nn.Parameter(1 * torch.randn(out_channels))
        self.r0_b1 = nn.Parameter(1 * torch.randn(out_channels))
        self.r0_b2 = nn.Parameter(1 * torch.randn(out_channels))

        # scalars
        self.kl = 0
        self.z = 0
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, out_channels, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, out_channels, num_transforms)

    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z
        zs, log_dets = self.z_flow(self.z)
        return zs, log_dets.squeeze()
    
    def kl_div(self):
        z2, log_det_q = self.sample_z()
        W_mean = self.weight_mu * z2.view(-1, 1, 1, 1) * self.alpha_q
        W_var = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2 * z2.view(-1, 1, 1, 1)**2)

        log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                  - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
        log_q = -log_det_q + log_q0

        act_mu = W_mean.view(-1, len(self.r0_c)) @ self.r0_c  # eq. (11)
        act_var = W_var.view(-1, len(self.r0_c)) @ self.r0_c ** 2  # eq. (12)

        # For convolutional layers, linear mappings empirically work better than
        # tanh. Hence no need for act = tanh(act). Christos Louizos
        # confirmed this in https://github.com/AMLab-Amsterdam/MNF_VBNN/issues/4
        # even though the paper states the use of tanh in conv layers.
        act = act_mu + act_var.sqrt() * torch.randn_like(act_var)

        # Mean and log variance of the auxiliary normal dist. r(z_T_b|W) in eq. 8.
        mean_r = self.r0_b1.outer(act).mean(-1)
        log_var_r = self.r0_b2.outer(act).mean(-1)

        z_b, log_det_r = self.r_flow(z2)
        log_rb = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * log_var_r
                  - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())).sum()
        log_r = log_det_r + log_rb
        
        kl_bias = (torch.log((self.bias_sigma_prior / (self.bias_sigma+torch.tensor(1e-45)))+torch.tensor(1e-45)) 
                       - 0.5 + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) 
                       / (2 * self.bias_sigma_prior ** 2 + torch.tensor(1e-45))).sum()

        kl_weight = (self.alpha_q * (torch.log((self.sigma_prior / (self.weight_sigma+torch.tensor(1e-45))) +torch.tensor(1e-45))
                                        - 0.5 + torch.log((self.alpha_q / (self.alpha_prior + torch.tensor(1e-45))) +torch.tensor(1e-45))
                                        + (self.weight_sigma ** 2 + (self.weight_mu*z2.view(-1, 1, 1, 1) - self.mu_prior) ** 2) / (
                                                2 * self.sigma_prior ** 2+torch.tensor(1e-45)))
                        + (1 - self.alpha_q) * torch.log(((1 - self.alpha_q) / (1 - self.alpha_prior + torch.tensor(1e-45)))+torch.tensor(1e-45))).sum()

        return  kl_bias + kl_weight + log_q - log_r

    def forward(self, input, ensemble=False, post_train=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        if post_train:
            self.alpha_q = (self.alpha_q.detach() > 0.5) * 1.
            self.alpha_prior[self.alpha_q.detach() < 0.5] = 0. # Only include priors that is inlcuded in the model
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        z, _ = self.sample_z()
        if self.training or ensemble:
            w_mean = self.weight_mu * z.view(-1, 1, 1, 1) * self.alpha_q
            w_var = self.alpha_q*(self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2 * z.view(-1, 1, 1, 1)**2)
            psi = F.conv2d(input, weight=w_mean, bias=self.bias_mu, stride=self.stride, padding=self.padding)
            delta = F.conv2d(input ** 2, weight=w_var, bias=self.bias_sigma ** 2, stride=self.stride, padding=self.padding)
            zeta = torch.randn_like(delta)
            activations = psi + torch.sqrt(delta) * zeta

        else: #median prob model
            w = torch.normal(self.weight_mu * z.view(-1,1,1,1),self.weight_sigma)
            bias = torch.normal(self.bias_mu,self.bias_sigma)
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = F.conv2d(input, weight=weight, bias=bias, stride=self.stride, padding=self.padding)
             
        return activations