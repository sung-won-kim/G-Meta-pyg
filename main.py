import os
from argument import config2string, parse_args
from utils import seed_everything
import torch
import yaml
from gmeta import gmeta_trainer

import warnings
warnings.filterwarnings(action='ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# limit CPU usage
torch.set_num_threads(2)


def main(args):

    embedder = gmeta_trainer(args, set_seed)

    (
        best_acc_train,
        best_f1_train,
        best_epoch_train,
        best_acc_valid,
        best_f1_valid,
        best_epoch_valid,
        best_acc_test,
        best_f1_test,
        best_epoch_test,
        test_acc_at_best_valid,
        test_f1_at_best_valid,
    ) = embedder.train()

    print("")
    print(
        f"# Best_Acc_Train : {best_acc_train}] at {best_epoch_train} epoch"
    )
    print(
        f"# Best_Acc_Valid : {best_acc_valid} at {best_epoch_valid} epoch"
    )
    print(
        f"# Best_Acc_Test : {best_acc_test}  at {best_epoch_test} epoch"
    )
    print(
        f"# Acc_Test_At_Best_Valid : {test_acc_at_best_valid} at {best_epoch_valid} epoch"
    )
    print("")


if __name__ == "__main__":
    args, unknown = parse_args()

    for set_seed in range(args.seed, args.seed + args.num_seed):
        seed_everything(set_seed)
        main(args)
