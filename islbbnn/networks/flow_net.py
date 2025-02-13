import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.flow_layers import BayesianLinear

class BayesianNetwork(nn.Module):
    def __init__(
            self, 
            dim, 
            p, 
            hidden_layers, 
            a_prior=0.05, 
            std_prior=2.5, 
            num_transforms=2, 
            classification=True, 
            n_classes=1, 
            act_func=F.sigmoid,
            lower_init_lambda=-10, 
            upper_init_lambda=-7,
            high_init_covariate_prob=False):
        '''
        TODO: Add option to select perfered loss self wanting to test another loss type 
        '''
        super().__init__()
        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func
        if high_init_covariate_prob:
            nr_var = p
        else:
            nr_var = None
        # set the architecture
        self.linears = nn.ModuleList([BayesianLinear(p, dim, a_prior=a_prior, num_transforms=num_transforms, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var)])
        self.linears.extend([BayesianLinear((dim+p), (dim), a_prior=a_prior, num_transforms=num_transforms, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var) for _ in range(hidden_layers-1)])
        self.linears.append(BayesianLinear((dim+p), n_classes, a_prior=a_prior, num_transforms=num_transforms, std_prior=std_prior, lower_init_lambda=lower_init_lambda, upper_init_lambda=upper_init_lambda, p=nr_var))
        if classification:
            if not self.multiclass: 
                self.loss = nn.BCELoss(reduction='sum') # Setup loss (Binary cross entropy as binary classification)
            else:
                self.loss = nn.NLLLoss(reduction='sum')
        else:
            self.loss = nn.MSELoss(reduction='sum')     # Setup loss (Mean Squared Error loss as regression problem)
            
    def forward(self, x, sample=False, ensemble=True, calculate_log_probs=False, post_train=False):
        '''
        x: 
            Input data
        ensemble:
            If True, then we will use the full model. If False, we will use the median prob model
        post_train:
            Train using the median probability model
        
        TODO: sample and calculate_log_probs are not used in flow_layers, but are used in lrt_layers
              Therefore, we should find a way s.t. we do not need to give things that are not used
              in order to use the networks. That is, make more general...
        '''
        x_input = x.view(-1, self.p)
        x = self.act(self.linears[0](x_input, ensemble, post_train))
        i = 1
        for l in self.linears[1:-1]:
            x = self.act(l(torch.cat((x, x_input),1), ensemble, post_train))
            i += 1

        if self.classification:
            if self.multiclass:
                out = F.log_softmax((self.linears[i](torch.cat((x, x_input),1), ensemble, post_train)), dim=1)
            else:
                out = torch.sigmoid(self.linears[i](torch.cat((x, x_input),1), ensemble, post_train))
        else:
            out = self.linears[i](torch.cat((x, x_input),1), ensemble, post_train)
        return out

    def forward_preact(self, x, sample=False, ensemble=False, calculate_log_probs=False, post_train=False):
        '''
        x: 
            Input data
        ensemble:
            If True, then we will use the full model. If False, we will use the median prob model
        post_train:
            Train using the median probability model
        
        TODO: sample and calculate_log_probs are not used in flow_layers, but are used in lrt_layers
              Therefore, we should find a way s.t. we do not need to give things that are not used
              in order to use the networks. That is, make more general...
        '''
        x_input = x.view(-1, self.p)
        x = self.act(self.linears[0](x_input, ensemble, post_train))
        i = 1
        for l in self.linears[1:-1]:
            x = self.act(l(torch.cat((x, x_input),1), ensemble, post_train))
            i += 1

        out = self.linears[i](torch.cat((x, x_input),1), ensemble, post_train)
        return out

    def kl(self):
        kl_sum = self.linears[0].kl_div()
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl_div()
        return kl_sum 
    
