import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pickle

valid_num_dic = {"Amazon_clothing": 17, "Amazon_eletronics": 36, "dblp": 27}


def load_data(dataset_source):
    n1s = []
    n2s = []
    for line in open("./data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split("\t")
        n1s.append(int(n1))
        n2s.append(int(n2))

    edges = torch.LongTensor([n1s, n2s])

    num_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_nodes, num_nodes))

    data_train = sio.loadmat("./data/{}_train.mat".format(dataset_source))
    train_class = list(
        set(data_train["Label"].reshape((1, len(data_train["Label"])))[0])
    )

    data_test = sio.loadmat("./data/{}_test.mat".format(dataset_source))
    class_list_test = list(
        set(data_test["Label"].reshape((1, len(data_test["Label"])))[0])
    )

    labels = np.zeros((num_nodes, 1))
    labels[data_train["Index"]] = data_train["Label"]
    labels[data_test["Index"]] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train["Index"]] = data_train["Attributes"].toarray()
    features[data_test["Index"]] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    class_list_valid = random.sample(
        train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return (
        edges,
        adj,
        features,
        labels,
        degree,
        class_list_train,
        class_list_valid,
        class_list_test,
        id_by_class,
    )


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average="weighted")
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def seed_everything(seed=0):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def task_generator_in_class(
    id_by_class, selected_class_list, n_way, k_shot, m_query
):  # id_unmasked 추가 가능
    # sample class indices
    class_selected = selected_class_list
    id_support = []
    id_query = []

    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    # return [0] (k-shot x n_way) support data id array
    #        [1] (n_query x n_way) query data id array
    #        [2] (n_way) selected class list
    return np.array(id_support), np.array(id_query), class_selected
