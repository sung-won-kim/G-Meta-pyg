import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Classifier
from copy import deepcopy
from conv import GCN
from tqdm import tqdm


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = n_support

    prototypes = input_cpu.view([n_classes, n_query, -1])
    prototypes = prototypes.mean(1)

    dists = euclidean_dist(input_cpu, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes


def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(y_t.shape[0]/n_classes)

    dists = euclidean_dist(input_cpu, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


class Meta(nn.Module):
    def __init__(self, args, config, features, labels):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.qry
        self.task_num = args.episodes  # 50
        self.update_step = args.update_step  # 5
        self.update_step_test = args.update_step_test  # 10
        self.features = features
        self.args = args
        self.labels = labels

        new_y = []
        for i in range(self.n_way):
            new_y += [i]*self.k_spt
        self.new_y = torch.LongTensor(new_y).to(args.device)

        new_qry_y = []
        for i in range(self.n_way):
            new_qry_y += [i]*self.k_qry
        self.new_qry_y = torch.LongTensor(new_qry_y).to(args.device)

        self.net = Classifier(config)
        self.net = self.net.to(args.device)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

        self.method = 'G-Meta'

    def forward_ProtoMAML(self, tasks):
        """
        b: number of tasks
        setsz: the size for each task

        # setsz subgraphs
        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of
        :param y_spt:   [b, setsz]
        # setsz subgraphs
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(tasks)
        querysz = self.k_qry
        losses_s = [0 for _ in range(self.update_step)]
        # losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        # __________
        # mini batch
        for i, task in enumerate(tasks):
            node_subgraph_spt_i, edge_subgraph_spt_i, edge_subgraph_spt_relabel_i, y_spt_i, center_node_relative_index_spt_i, node_subgraph_qry_i, edge_subgraph_qry_i, edge_subgraph_qry_relabel_i, y_qry_i, center_node_relative_index_qry_i = task

            feat_spt = self.features[node_subgraph_spt_i]
            feat_qry = self.features[node_subgraph_qry_i]

            logits, _ = self.net(node_subgraph_spt_i.to(
                self.device), edge_subgraph_spt_relabel_i.to(self.device), center_node_relative_index_spt_i.to(self.device), feat_spt, vars=None)

            logits_spt = logits[center_node_relative_index_spt_i]

            loss, _, prototypes = proto_loss_spt(
                logits_spt, y_spt_i, self.k_spt)

            losses_s[0] += loss

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, self.net.parameters())

                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q, _ = self.net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, fast_weights)

                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(node_subgraph_spt_i.to(
                    self.device), edge_subgraph_spt_relabel_i.to(self.device), center_node_relative_index_spt_i.to(self.device), feat_spt, fast_weights)

                logits_spt = logits[center_node_relative_index_spt_i]

                loss, _, prototypes = proto_loss_spt(
                    logits_spt, y_spt_i, self.k_spt)

                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(
                    loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)

                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        if torch.isnan(loss_q):
            pass
        else:
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        accs = np.array(corrects) / (task_num)

        return accs, loss_q

    def finetunning_ProtoMAML(self, tasks):

        task_num = len(tasks)
        querysz = self.k_qry
        losses_s = [0 for _ in range(self.update_step_test)]
        # losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(self.update_step_test + 1)]
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # __________
        # mini batch

        for i, task in enumerate(tasks):
            # finetunning on the copied model instead of self.net
            net = deepcopy(self.net)
            node_subgraph_spt_i, edge_subgraph_spt_i, edge_subgraph_spt_relabel_i, y_spt_i, center_node_relative_index_spt_i, node_subgraph_qry_i, edge_subgraph_qry_i, edge_subgraph_qry_relabel_i, y_qry_i, center_node_relative_index_qry_i = task

            feat_spt = self.features[node_subgraph_spt_i]
            feat_qry = self.features[node_subgraph_qry_i]

            # 1. run the i-th task and compute loss for k=0
            logits, _ = net(node_subgraph_spt_i.to(
                self.device), edge_subgraph_spt_relabel_i.to(self.device), center_node_relative_index_spt_i.to(self.device), feat_spt, vars=None)

            logits_spt = logits[center_node_relative_index_spt_i]

            loss, _, prototypes = proto_loss_spt(
                logits_spt, y_spt_i, self.k_spt)

            losses_s[0] += loss

            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, net.parameters())

                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, fast_weights)
                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = net(node_subgraph_spt_i.to(
                    self.device), edge_subgraph_spt_relabel_i.to(self.device), center_node_relative_index_spt_i.to(self.device), feat_spt, fast_weights)

                logits_spt = logits[center_node_relative_index_spt_i]

                loss, _, prototypes = proto_loss_spt(
                    logits_spt, y_spt_i, self.k_spt)

                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(
                    loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q, _ = net(node_subgraph_qry_i.to(
                    self.device), edge_subgraph_qry_relabel_i.to(self.device), center_node_relative_index_qry_i.to(self.device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                logits_qry = logits_q[center_node_relative_index_qry_i]

                loss_q, acc_q = proto_loss_qry(logits_qry, y_qry_i, prototypes)

                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

            del net

        loss_q = losses_q[-1] / task_num

        accs = np.array(corrects) / (task_num)

        return accs, loss_q

    def forward(self, tasks):
        if self.method == 'G-Meta':
            accs = self.forward_ProtoMAML(
                tasks)
        return accs

    def finetunning(self, tasks):
        if self.method == 'G-Meta':
            accs = self.finetunning_ProtoMAML(
                tasks)
        return accs
