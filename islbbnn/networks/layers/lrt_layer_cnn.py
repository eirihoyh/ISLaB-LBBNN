import torch
import torch.nn as nn
import torch.nn.functional as F
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BayesianConv2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size,
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

        # scalars
        self.kl = 0

    def forward(self, input, ensemble=False, sample=False, calculate_log_probs=False, post_train=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        if post_train:
            self.alpha_q = (self.alpha_q.detach() > 0.5) * 1
            self.alpha_prior[self.alpha_q.detach() < 0.5] = 0. 
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or ensemble:
            w_mean = self.weight_mu * self.alpha_q
            w_var = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            psi = F.conv2d(input, weight=w_mean, bias=self.bias_mu, stride=self.stride, padding=self.padding)
            delta = F.conv2d(input ** 2, weight=w_var, bias=self.bias_sigma ** 2, stride=self.stride, padding=self.padding)
            zeta = torch.randn_like(delta)
            activations = psi + torch.sqrt(delta) * zeta

        else: # median probability model
            if sample:
                w = torch.normal(self.weight_mu,self.weight_sigma)
                bias = torch.normal(self.bias_mu,self.bias_sigma)
            else:
                w = self.weight_mu
                bias = self.bias_mu
            g = (self.alpha_q.detach() > 0.5) * 1.
            weight = w * g
            activations = F.conv2d(input, weight=weight, bias=bias, stride=self.stride, padding=self.padding)

            if calculate_log_probs:
                self.alpha_q = g
        

        if self.training or calculate_log_probs:

            kl_bias = (torch.log((self.bias_sigma_prior / (self.bias_sigma+torch.tensor(1e-45)))+torch.tensor(1e-45)) 
                       - 0.5 + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) 
                       / (2 * self.bias_sigma_prior ** 2 + torch.tensor(1e-45))).sum()

            kl_weight = (self.alpha_q * (torch.log((self.sigma_prior / (self.weight_sigma+torch.tensor(1e-45))) +torch.tensor(1e-45))
                                         - 0.5 + torch.log((self.alpha_q / (self.alpha_prior + torch.tensor(1e-45))) +torch.tensor(1e-45))
                                         + (self.weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2) / (
                                                 2 * self.sigma_prior ** 2+torch.tensor(1e-45)))
                         + (1 - self.alpha_q) * torch.log(((1 - self.alpha_q) / (1 - self.alpha_prior + torch.tensor(1e-45)))+torch.tensor(1e-45))).sum()

            self.kl = kl_bias + kl_weight

        else:
            self.kl = 0
        # propogate
        return activations