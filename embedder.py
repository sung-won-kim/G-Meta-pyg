import torch
import torch.nn as nn
import os
from torch_geometric.data import DataLoader
from argument import config2string
from tensorboardX import SummaryWriter

from sklearn.decomposition import PCA
from utils import *

# _____________
# embedder init


class embedder(nn.Module):
    def __init__(self, args, set_seed):
        super().__init__()

        self.args = args
        self.set_seed = set_seed
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available(
        ) else 'cpu'
        torch.cuda.set_device(self.device)

        # _________________
        # episodic training

        # load dataset
        self.edges, self.adj, self.features, self.labels, self.degrees, self.class_list_train, self.class_list_valid, self.class_list_test, self.id_by_class = load_data(
            args.dataset)

        self.edges = self.edges.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.degrees = self.degrees.to(self.device)

        self.n_way = args.way
        self.k_shot = args.shot
        self.n_query = args.qry
