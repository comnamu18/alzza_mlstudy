import numpy as np
import random
import torch

# https://hoya012.github.io/blog/reproducible_pytorch/ 참고

def set_seed(seed : int):
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
