import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, args):
        super(mlp, self).__init__()
        modules = []
        prev = args.in_features      
        for h in args.hidden:
            modules.append(nn.Linear(in_features=prev, out_features=h, bias=True))
            modules.append(nn.ReLU())
            prev = h
        modules.append(nn.Linear(in_features=prev, out_features=args.out_features, bias=True))

        self.net = nn.Sequential(*modules)
        
        # initialize weight and bias
        for layer in self.net:
            if type(layer) is nn.Linear:
                nn.init.normal_(layer.weight, mean=args.init_mean, std=args.init_std)
                nn.init.normal_(layer.bias, mean=args.init_mean, std=args.init_std)
    
    def forward(self, x):
        return self.net(x)