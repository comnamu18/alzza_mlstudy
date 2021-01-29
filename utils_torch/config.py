import argparse

def flower_mlp_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--report', type=int, default=10)
    parser.add_argument('--in_features', type=int, default=30000)
    parser.add_argument('--out_features', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--init_mean', type=float, default=0.)
    parser.add_argument('--init_std', type=float, default=0.003)
    args = parser.parse_args()

    if args.hidden is None:
        args.hidden = []
    else:
        args.hidden = [int(h.strip()) for h in args.hidden.split(',')]
    
    return args