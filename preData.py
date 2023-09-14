import os
import torch
import dgl
import numpy as np
from sklearn.model_selection import KFold
from utils import *

# ***************************载入图的相似性特征开始**************************
def similarity_feature_process(args):
    similarity_feature = {}
    path = args.path
    device = args.device

    "miRNA functional sim"
    rna_func_sim = np.loadtxt(os.path.join(path, 'm_fs.csv'), delimiter=',', dtype=float)
    rna_func_sim = torch.tensor(rna_func_sim, device=device).to(torch.float32)
    rna_func_edge_index = get_edge_index(rna_func_sim, device)
    g_mm_f = dgl.graph((rna_func_edge_index[0], rna_func_edge_index[1]))
    similarity_feature['mm_f'] = {'Data_M': rna_func_sim, 'edges': rna_func_edge_index, 'g': g_mm_f}

    "miRNA Gaussian sim"
    rna_Gaus_sim = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float)
    rna_Gaus_sim = torch.tensor(rna_Gaus_sim, device=device).to(torch.float32)
    rna_Gaus_edge_index = get_edge_index(rna_Gaus_sim, device)
    g_mm_g = dgl.graph((rna_Gaus_edge_index[0], rna_Gaus_edge_index[1]))
    similarity_feature['mm_g'] = {'Data_M': rna_Gaus_sim, 'edges': rna_Gaus_edge_index, 'g': g_mm_g}

    miRNA_similarity = integ_similarity(rna_func_sim,rna_Gaus_sim)
    similarity_feature['m_s'] = {'Data_M': miRNA_similarity}

    "disease semantic sim"
    dis_semantic_sim = np.loadtxt(os.path.join(path, 'd_ss.csv'), delimiter=',', dtype=float)
    dis_semantic_sim = torch.tensor(dis_semantic_sim, device=device).to(torch.float32)
    dis_sem_edge_index = get_edge_index(dis_semantic_sim, device)
    g_dd_s = dgl.graph((dis_sem_edge_index[0], dis_sem_edge_index[1]))
    similarity_feature['dd_s'] = {'Data_M': dis_semantic_sim, 'edges': dis_sem_edge_index,'g':g_dd_s}

    "disease Gaussian sim"
    dis_Gaus_sim = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float)
    dis_Gaus_sim = torch.tensor(dis_Gaus_sim, device=device).to(torch.float32)
    dis_Gaus_edge_index = get_edge_index(dis_Gaus_sim, device)
    g_dd_g = dgl.graph((dis_Gaus_edge_index[0], dis_Gaus_edge_index[1]))
    similarity_feature['dd_g'] = {'Data_M': dis_Gaus_sim, 'edges': dis_Gaus_edge_index,'g':g_dd_g}

    disease_similarity = integ_similarity(dis_semantic_sim, dis_Gaus_sim)
    similarity_feature['d_s'] = {'Data_M': disease_similarity}
    return similarity_feature

# ***************************载入图的边数据开始**************************
def load_fold_data(args):
    path = args.path
    kfolds = args.kfolds
    edge_idx_dict = dict()
    g = dict()
    md_matrix = np.loadtxt(os.path.join(path + '/m_d.csv'), dtype=int, delimiter=',')
    edge_idx_dict['true_md'] = md_matrix  ##*853*591矩阵中一共12446个1

    # get the edge of positives samples
    rng = np.random.default_rng()

    pos_samples = np.where(md_matrix == 1)
    pos_samples = (pos_samples[0], pos_samples[1] + md_matrix.shape[0])
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)


    train_pos_edges = pos_samples_shuffled  # 11201正 90%

    neg_samples = np.where(md_matrix == 0)
    neg_samples = (neg_samples[0], neg_samples[1] + md_matrix.shape[0])
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    train_neg_edges = neg_samples_shuffled

    # 用于模型测试的标签

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)

    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T

    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(train_pos_edges):
        train_idx.append(train_index)
        valid_idx.append(valid_index)

    for i in range(kfolds):
        edges_train_pos, edges_valid_pos = train_pos_edges[train_idx[i]], train_pos_edges[valid_idx[i]]
        fold_train_pos80 = edges_train_pos.T
        fold_valid_pos20 = edges_valid_pos.T

        edges_train_neg, edges_valid_neg = train_neg_edges[train_idx[i]], train_neg_edges[valid_idx[i]]
        fold_train_neg80 = edges_train_neg.T
        fold_valid_neg20 = edges_valid_neg.T

        fold_100p_100n = np.hstack(
            (np.hstack((fold_train_pos80, fold_valid_pos20)), np.hstack((fold_train_neg80, fold_valid_neg20))))
        fold_train_edges_80p_80n = np.hstack((fold_train_pos80, fold_train_neg80))
        fold_train_label_80p_80n = np.hstack((np.ones(fold_train_pos80.shape[1]), np.zeros(fold_train_neg80.shape[1])))
        fold_valid_edges_20p_20n = np.hstack((fold_valid_pos20, fold_valid_neg20))
        fold_valid_label_20p_20n = np.hstack((np.ones(fold_valid_pos20.shape[1]), np.zeros(fold_valid_neg20.shape[1])))


        edge_idx_dict[str(i)] = {}
        g[str(i)] = {}
        #可能用到的100
        edge_idx_dict[str(i)]["fold_100p_100n"] = torch.tensor(fold_100p_100n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_100p_100n"] = dgl.graph(
            (fold_100p_100n[0], fold_100p_100n[1])).to(device=args.device)
        #训练用的80
        edge_idx_dict[str(i)]["fold_train_edges_80p_80n"] = torch.tensor(fold_train_edges_80p_80n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_train_edges_80p_80n"] = dgl.graph(
            (fold_train_edges_80p_80n[0], fold_train_edges_80p_80n[1])).to(device=args.device)

        edge_idx_dict[str(i)]["fold_train_label_80p_80n"] = torch.tensor(fold_train_label_80p_80n).to(torch.float32).to(
            device=args.device)
        #验证用的20
        edge_idx_dict[str(i)]["fold_valid_edges_20p_20n"] = torch.tensor(fold_valid_edges_20p_20n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_valid_edges_20p_20n"] = dgl.graph((fold_valid_edges_20p_20n[0], fold_valid_edges_20p_20n[1])).to(device=args.device)
        edge_idx_dict[str(i)]["fold_valid_label_20p_20n"] = torch.tensor(fold_valid_label_20p_20n).to(torch.float32).to(
            device=args.device)

    return edge_idx_dict, g