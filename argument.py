import argparse
from ast import parse
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str,
                        default="Amazon_clothing")  # "Amazon_clothing" , "Amazon_eletronics",  "dblp", "arxiv"
    parser.add_argument("--way", type=int, default=3)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--qry", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)

    parser.add_argument('--meta_lr', type=float,
                        help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float,
                        help='task-level inner update learning rate', default=1e-3)
    parser.add_argument('--update_step', type=int,
                        help='task-level inner update steps', default=10)
    parser.add_argument('--update_step_test', type=int,
                        help='update steps for finetunning', default=10)
    parser.add_argument('--hidden_dim', type=int,
                        help='hidden dim', default=50)
    parser.add_argument("--h", default=1, type=int,
                        required=False, help="neighborhood size")

    return parser.parse_known_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ""
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in [
            "device",
            "patience",
            "epochs"
        ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]
