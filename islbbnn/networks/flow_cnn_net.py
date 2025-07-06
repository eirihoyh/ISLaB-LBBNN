import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.flow_layers import BayesianLinear
from layers.flow_layer_cnn import BayesianConv2d

def compute_output_dimensions(
    n_layers, 
    input_width, 
    input_height, 
    kernel_size, 
    stride,
    padding):
    """
    Computes the output width and height for output layer in a CNN.

    Parameters:
    - n_layers: int
    - input_width: int
    - input_height: int
    - kernel_size: int
    - padding: int
    - stride: int
    
    

    Returns:
    - output_width, output_height for final layer
    """
    width, height = input_width, input_height
    for _ in range(n_layers):
        width = (width - kernel_size + 2 * padding) // stride + 1
        height = (height - kernel_size + 2 * padding) // stride + 1

    return width, height


class BCNN(nn.Module):
    def __init__(
            self, 
            init_in_channels,
            out_channel_list,
            kernel_size,
            stride,
            padding,
            p1, # number of pixel rows
            p2, # number of pixel columns
            dim,
            hidden_layers,
            a_prior=0.05, 
            std_prior=2.5, 
            num_transforms=2,
            n_classes=1, 
            act_func=F.sigmoid, 
            lower_init_lambda=-10, 
            upper_init_lambda=-7,
            high_init_covariate_prob=False
            ):
        super().__init__()
        self.init_in_channels = init_in_channels
        self.p1 = p1
        self.p2 = p2
        self.p = p1*p2*init_in_channels
        self.multiclass = n_classes > 1
        self.act = act_func
        if high_init_covariate_prob:
            nr_var = self.p
        else:
            nr_var = None
        # set the architecture

        # cnn part
        self.convs = nn.ModuleList([BayesianConv2d(in_channels=init_in_channels, out_channels=out_channel_list[0], num_transforms=num_transforms, kernel_size=kernel_size, stride=stride, padding=padding, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, a_prior=a_prior, std_prior=std_prior)])
        self.convs.extend([BayesianConv2d(in_channels=out_channel_list[i], num_transforms=num_transforms, out_channels=out_channel_list[i+1], kernel_size=kernel_size, stride=stride, padding=padding, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, a_prior=a_prior, std_prior=std_prior) for i in range(len(out_channel_list)-1)])
        
        width_out, height_out = compute_output_dimensions(len(self.convs), p1, p2, kernel_size, stride, padding)
        cnn_output_dim = out_channel_list[-1]*width_out*height_out
        # ISLaB part
        self.linears = nn.ModuleList([BayesianLinear((cnn_output_dim+self.p), dim, num_transforms=num_transforms, a_prior=a_prior, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var)])
        self.linears.extend([BayesianLinear((dim+self.p), (dim), num_transforms=num_transforms, a_prior=a_prior, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var) for _ in range(hidden_layers-1)])
        self.linears.append(BayesianLinear((dim+self.p), n_classes, num_transforms=num_transforms, a_prior=a_prior, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var))
        if not self.multiclass: 
            self.loss = nn.BCELoss(reduction='sum') # Setup loss (Binary cross entropy as binary classification)
        else:
            self.loss = nn.NLLLoss(reduction='sum')


    def forward(self, x, sample=False, ensemble=True, calculate_log_probs=False, post_train=False):
        x_input = x
        x = x.view(-1, self.init_in_channels, self.p1, self.p2)
        for conv in self.convs:
            x = self.act(conv(x, ensemble, post_train))
        
        x = x.flatten(1)

        for l in self.linears[:-1]:
            x = self.act(l(torch.cat((x, x_input),1), ensemble, post_train))

        if self.multiclass:
            out = F.log_softmax(self.linears[-1](torch.cat((x, x_input),1), ensemble, post_train), dim=1)
        else:
            out = torch.sigmoid(self.linears[-1](torch.cat((x, x_input),1), ensemble, post_train))
        
        return out
    
    def forward_preact(self, x, sample=False, ensemble=True, calculate_log_probs=False, post_train=False):
        x_input = x
        x = x.view(-1, self.init_in_channels, self.p1, self.p2)
        for conv in self.convs:
            x = self.act(conv(x, ensemble, post_train))
        
        x = x.flatten(1)

        for l in self.linears[:-1]:
            x = self.act(l(torch.cat((x, x_input),1), ensemble, post_train))

        out = self.linears[-1](torch.cat((x, x_input),1), ensemble, post_train)
        
        return out

    def kl(self):
        kl_sum = 0
        for conv in self.convs:
            kl_sum = kl_sum + conv.kl_div()


        for l in self.linears:
            kl_sum = kl_sum + l.kl_div()

        return kl_sum