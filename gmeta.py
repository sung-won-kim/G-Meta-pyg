
from embedder import embedder
from tqdm import tqdm
from utils import *
from argument import config2string

import torch

from meta import Meta
import copy
from torch_geometric.utils import k_hop_subgraph


class gmeta_trainer(embedder):
    def __init__(self, args, conf, set_seed):
        embedder.__init__(self, args, conf, set_seed)
        self.total_class = len(np.unique(np.array(self.labels.cpu())))
        print('There are {} classes '.format(self.total_class))
        self.labels_num = args.way
        self.feat = [self.features]

        self.config = [
            ('GraphConv', [self.features.shape[1], args.hidden_dim])]

        if args.h > 1:
            self.config = self.config + \
                [('GraphConv', [args.hidden_dim, args.hidden_dim])] * (args.h - 1)

        self.config = self.config + \
            [('Linear', [args.hidden_dim, self.labels_num])]

        self.maml = Meta(args, self.config, self.features,
                         self.labels).to(self.device)

        tmp = filter(lambda x: x.requires_grad, self.maml.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(self.maml)
        print('Total trainable tensors:', num)

        max_acc = 0
        model_max = copy.deepcopy(self.maml)

        self.config_str = config2string(args)
        self.args = args
        self.set_seed = set_seed
        self.patience = args.patience

    def db_batch(self, class_list):
        train_tasks = []

        for episode in range(self.args.episodes):

            class_selected = random.sample(
                class_list, self.n_way)

            id_support, id_query, class_selected = task_generator_in_class(
                self.id_by_class, class_selected, self.n_way, self.k_shot, self.n_query)

            # _______________________
            # support set subgraphing
            spt_sub = k_hop_subgraph(node_idx=torch.LongTensor(
                id_support), num_hops=self.args.h, edge_index=self.adj._indices())  # (subgraph 속 node_id, subgraph의 edge_index, 그 안에서 중심 노드의 순서 index, edge_mask)
            node_subgraph_spt = spt_sub[0]
            edge_subgraph_spt = spt_sub[1]
            center_node_relative_index_spt = spt_sub[2]

            spt_sub_relabel = k_hop_subgraph(node_idx=torch.LongTensor(
                id_support), num_hops=self.args.h, edge_index=self.adj._indices(), relabel_nodes=True)  # (subgraph 속 node_id, subgraph의 edge_index, 그 안에서 중심 노드의 순서 index, edge_mask)
            edge_subgraph_spt_relabel = spt_sub_relabel[1]

            # _____________________
            # query set subgraphing
            qry_sub = k_hop_subgraph(node_idx=torch.LongTensor(
                id_query), num_hops=self.args.h, edge_index=self.adj._indices())  # (subgraph 속 node_id, subgraph의 edge_index, 그 안에서 중심 노드의 순서 index, edge_mask)
            node_subgraph_qry = qry_sub[0]
            edge_subgraph_qry = qry_sub[1]
            center_node_relative_index_qry = qry_sub[2]

            spt_qry_relabel = k_hop_subgraph(node_idx=torch.LongTensor(
                id_query), num_hops=self.args.h, edge_index=self.adj._indices(), relabel_nodes=True)  # (subgraph 속 node_id, subgraph의 edge_index, 그 안에서 중심 노드의 순서 index, edge_mask)
            edge_subgraph_qry_relabel = spt_qry_relabel[1]

            y_spt = self.labels[id_support]
            y_qry = self.labels[id_query]

            train_tasks.append([node_subgraph_spt, edge_subgraph_spt, edge_subgraph_spt_relabel,
                                y_spt, center_node_relative_index_spt, node_subgraph_qry, edge_subgraph_qry, edge_subgraph_qry_relabel, y_qry, center_node_relative_index_qry])

        return train_tasks

    def train(self):

        # _____________
        # Best Accuracy
        best_acc_train = 0
        best_f1_train = 0
        best_epoch_train = 0
        best_acc_valid = 0
        best_f1_valid = 0
        best_epoch_valid = 0
        best_acc_test = 0
        best_f1_test = 0
        best_epoch_test = 0
        test_f1_at_best_valid = 0

        best_model = copy.deepcopy(self.maml)

        self.maml.train()

        for epoch in tqdm(range(self.args.epochs)):
            train_tasks = self.db_batch(self.class_list_train)
            valid_tasks = self.db_batch(self.class_list_valid)
            test_tasks = self.db_batch(self.class_list_test)

            accs_train, loss_train_epoch = self.maml(train_tasks)
            acc_train_epoch = accs_train.mean()

            # ____________________
            # Validation per epoch
            accs_valid, loss_valid_epoch = self.maml.finetunning(valid_tasks)
            acc_valid_epoch = accs_valid.mean()

            # ______________
            # Test per epoch
            accs_test, loss_test_epoch = self.maml.finetunning(test_tasks)
            acc_test_epoch = accs_test.mean()

            print(
                f"loss_train_epoch : {loss_train_epoch.item()}")
            print(f"acc_train : {acc_train_epoch}")

            if best_acc_train < acc_train_epoch:
                best_acc_train = acc_train_epoch
                best_epoch_train = epoch

            if best_acc_valid < acc_valid_epoch:
                current_patience = 0
                best_acc_valid = acc_valid_epoch
                best_epoch_valid = epoch

            print(f"acc_valid : {acc_valid_epoch}")

            if acc_valid_epoch == best_acc_valid:
                test_acc_at_best_valid = acc_test_epoch

            if best_acc_test < acc_test_epoch:
                best_acc_test = acc_test_epoch
                best_epoch_test = epoch

            print(f"acc_test : {acc_test_epoch}")

            print("")
            print(f"# Current Settings : {self.config_str}")
            print(
                f"# Best_Acc_Train : {best_acc_train} at {best_epoch_train} epoch"
            )
            print(
                f"# Best_Acc_Valid : {best_acc_valid} at {best_epoch_valid} epoch"
            )
            print(
                f"# Best_Acc_Test : {best_acc_test} at {best_epoch_test} epoch"
            )
            print(
                f"# Test_At_Best_Valid : {test_acc_at_best_valid} at {best_epoch_valid} epoch"
            )

            if acc_valid_epoch < best_acc_valid:
                current_patience += 1
                print(
                    f'current_patience : {current_patience} / {self.patience}')
                if current_patience == self.patience:
                    break

        return best_acc_train, best_f1_train, best_epoch_train, best_acc_valid, best_f1_valid, best_epoch_valid, best_acc_test, best_f1_test, best_epoch_test, test_acc_at_best_valid, test_f1_at_best_valid
