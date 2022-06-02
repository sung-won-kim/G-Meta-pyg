import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import csv
import random
import pickle
from torch.utils.data import DataLoader
import networkx as nx
import itertools


class Subgraphs(Dataset):
    def __init__(self, root, mode, subgraph2label, n_way, k_shot, k_query, batchsz, args, adjs, h):
        self.batchsz = batchsz  # episodes num
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.k_query = k_query  # for query set
        self.setsz = self.n_way * self.k_shot  # num of samples per support set
        # number of samples per set for evaluation
        self.querysz = self.n_way * self.k_query
        self.h = h  # number of h hops
        self.sample_nodes = args.sample_nodes
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, %d-hops' % (
            mode, batchsz, n_way, k_shot, k_query, h))

        # load subgraph list if preprocessed
        self.subgraph2label = subgraph2label

        if args.link_pred_mode == 'True':
            self.link_pred_mode = True
        else:
            self.link_pred_mode = False

        if self.link_pred_mode:
            dictLabels_spt, dictGraphs_spt, dictGraphsLabels_spt = self.loadCSV(
                os.path.join(root, mode + '_spt.csv'))
            dictLabels_qry, dictGraphs_qry, dictGraphsLabels_qry = self.loadCSV(
                os.path.join(root, mode + '_qry.csv'))
            dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(
                os.path.join(root, mode + '.csv'))  # csv path
        else:
            dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(
                os.path.join(root, mode + '.csv'))  # csv path

        self.task_setup = args.task_setup  # Disjoint

        self.G = []

        for i in adjs:
            self.G.append(i)

        self.subgraphs = {}

        if self.task_setup == 'Disjoint':
            self.data = []

            # i : enumerate, k : label, v : subgraph's index
            for i, (k, v) in enumerate(dictLabels.items()):
                # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
                self.data.append(v)
            self.cls_num = len(self.data)

            self.create_batch_disjoint(self.batchsz)

    def loadCSV(self, csvf):
        dictGraphsLabels = {}
        dictLabels = {}
        dictGraphs = {}

        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                g_idx = int(filename.split('_')[0])
                label = row[2]
                # append filename to current label

                if g_idx in dictGraphs.keys():
                    dictGraphs[g_idx].append(filename)
                else:
                    dictGraphs[g_idx] = [filename]
                    dictGraphsLabels[g_idx] = {}

                if label in dictGraphsLabels[g_idx].keys():
                    dictGraphsLabels[g_idx][label].append(filename)
                else:
                    dictGraphsLabels[g_idx][label] = [filename]

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, dictGraphs, dictGraphsLabels

    def create_batch_disjoint(self, batchsz):
        """
        create the entire set of batches of tasks for disjoint label setting, indepedent of # of graphs.
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            # print(self.cls_num)
            # print(self.n_way)
            selected_cls = np.random.choice(
                self.cls_num, self.n_way, False)  # no duplicate  ex) [4,3,5]
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:

                # 2. select k_shot + k_query for each class
                selected_subgraphs_idx = np.random.choice(
                    len(self.data[cls]), self.k_shot + self.k_query, False)

                np.random.shuffle(selected_subgraphs_idx)
                indexDtrain = np.array(
                    selected_subgraphs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(
                    selected_subgraphs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * n_way)] numbers of subgraphs
            # append set to current sets
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)  # append sets to current sets

    # helper to generate subgraphs on the fly.

    def generate_subgraph(self, G, i, item):
        if item in self.subgraphs:
            return self.subgraphs[item]
        else:
            # instead of calculating shortest distance, we find the following ways to get subgraphs are quicker
            if self.h == 2:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(
                    list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
            elif self.h == 1:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [i]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in G.in_edges(i)[0]]
                       for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(
                    *n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
            if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(
                    h_hops_neighbor, self.sample_nodes, replace=False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))

            sub = G.subgraph(h_hops_neighbor)
            h_c = list(sub.parent_nid.numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.subgraphs[item] = (sub, dict_[i], h_c)

            return sub, dict_[i], h_c
            '''
            sub : 중심 노드에 대한 subgraph
            h_c : subgraph를 구성하는 노드들의 실제 id list [12,34,56,78]
            dict_[i] : subgraph를 구성하는 node id list 중에서 target node가 몇 번째 인덱스인지 ex) 34번 노드가 중심이라면 1이 담긴다.
            '''

    def __getitem__(self, index):
        """
        get one task. support_x_batch[index], query_x_batch[index]

        """

        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                for sublist in self.support_x_batch[index] for item in sublist]

        support_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]  # 이건 그냥 graph 1개만 쓰는 task 이므로 다 0으로 나옴 [0, 0, 0, 0, 0, 0, 0, 0, 0] 3way * 3shot = 9개

        support_x = [i for i, j, k in info]  # 3way * 3shot = 9개의 subgraph list
        support_y = np.array([self.subgraph2label[item]
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)  # 3way * 3shot = 9개의 subgraph label (중심 노드의 label인듯)

        # subgraph를 이루는 실제 노드 index list 중에서 몇 번째 노드가 중심 노드인지 정보
        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        # subgraph를 이루는 실제 노드 index list
        support_node_idx = [k for i, j, k in info]

        info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                for sublist in self.query_x_batch[index] for item in sublist]

        query_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                           for sublist in self.query_x_batch[index] for item in sublist]  # 24query * 3way = 72개의 query에 대한 graph label => 1개의 graph에서 뽑는 task이므로 그냥 다 0이다.

        # 24qry * 3way = 72개의 query를 subgraph로 구성함.
        query_x = [i for i, j, k in info]
        query_y = np.array([self.subgraph2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        if self.task_setup == 'Disjoint':
            unique = np.unique(support_y)
            random.shuffle(unique)
            # relative means the label ranges from 0 to n-way
            support_y_relative = np.zeros(self.setsz)
            query_y_relative = np.zeros(self.querysz)
            for idx, l in enumerate(unique):
                support_y_relative[support_y == l] = idx
                query_y_relative[query_y == l] = idx
             # this is a set of subgraphs for one task.
            # 9개의 support x를 1개의 batch로 묶어주네
            batched_graph_spt = dgl.batch(support_x)
            batched_graph_qry = dgl.batch(query_x)
            '''
            batched_graph_spt.batch_size : 9
            batched_graph_spt.batch_num_nodes : 
            '''

            return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(query_y_relative), torch.LongTensor(support_center), torch.LongTensor(query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(
        list, zip(*samples))

    return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx
