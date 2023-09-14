import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.batch import unbatch
from dgl.transforms import shortest_dist
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import JumpingKnowledge

from param import *


class DegreeEncoder(nn.Module):
    def __init__(self, max_degree, embedding_dim):
        super(DegreeEncoder, self).__init__()
        self.encoder1 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.encoder2 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.max_degree = max_degree

    def forward(self, g):
        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)
        degree_embedding = self.encoder1(in_degree) + self.encoder2(out_degree)
        return degree_embedding


class SpatialEncoder(nn.Module):
    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )

    def forward(self, g):
        device = g.device
        g_list = unbatch(g)
        max_num_nodes = th.max(g.batch_num_nodes())
        spatial_encoding = th.zeros(
            len(g_list), max_num_nodes, max_num_nodes, self.num_heads
        ).to(device)

        for i, ubg in enumerate(g_list):
            num_nodes = ubg.num_nodes()
            dist = (
                    th.clamp(
                        shortest_dist(ubg, root=None, return_paths=False),
                        min=-1,
                        max=self.max_dist,
                    )
                    + 1
            )
            dist_embedding = self.embedding_table(dist)
            spatial_encoding[i, :num_nodes, :num_nodes] = dist_embedding
        return spatial_encoding


class BiasedMHA(nn.Module):

    def __init__(
            self,
            feat_size,
            num_heads,
            bias=True,
            attn_bias_type="add",
            attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
                self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.u_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.GLU.weight, gain=2 ** -0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):

        q_h = self.q_proj(ndata).transpose(0, 1)
        k_h = self.k_proj(ndata).transpose(0, 1)
        v_h = self.v_proj(ndata).transpose(0, 1)

        bsz, N, _ = ndata.shape
        q_h = (
                q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                * self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(
            1, 2, 0
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )

        attn_weights = (
            th.bmm(q_h, k_h)
                .transpose(0, 2)
                .reshape(N, N, bsz, self.num_heads)
                .transpose(0, 2)
        )

        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")
        attn_weights = F.softmax(
            attn_weights.transpose(0, 2)
                .reshape(N, N, bsz * self.num_heads)
                .transpose(0, 2),
            dim=2,
        )

        attn_weights = self.dropout(attn_weights)

        attn = th.bmm(attn_weights, v_h).transpose(0, 1)

        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(0, 1)
        )
        return attn


class GraphormerLayer(nn.Module):

    def __init__(
            self,
            feat_size,
            hidden_size,
            num_heads,
            attn_bias_type="add",
            norm_first=False,
            dropout=0.1,
            attn_dropout=0.1,
            activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMHA(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):

        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(args.hidden, args.hidden // 2)
        self.bn1 = nn.BatchNorm1d(args.hidden // 2)
        self.dropout1 = nn.Dropout(args.MLPDropout)
        self.fc2 = nn.Linear(args.hidden // 2, args.hidden // 4)
        self.bn2 = nn.BatchNorm1d(args.hidden // 4)
        self.dropout2 = nn.Dropout(args.MLPDropout)
        self.fc3 = nn.Linear(args.hidden // 4, args.hidden // 8)
        self.bn3 = nn.BatchNorm1d(args.hidden // 8)
        self.dropout3 = nn.Dropout(args.MLPDropout)
        self.fc4 = nn.Linear(args.hidden // 8, 1)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.tanh(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.tanh(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# 最大
class Feature_Weight(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(Feature_Weight, self).__init__()
        sumN = n_rna + n_dis
        self.GAT_Weight = Parameter(torch.randn(sumN, args.hidden))
        self.Trans_Wei = Parameter(torch.randn(sumN, args.hidden))
        self.fc = nn.Linear(args.hidden * 2, args.hidden)

    def forward(self, G, T):

        # ************最大**************
        Feature_Finnally, _ = torch.max(torch.stack([G, T]), dim=0)

        return Feature_Finnally


class GCN_GAT(nn.Module):
    def __init__(self, args):
        super(GCN_GAT, self).__init__()
        self.ReLU = nn.ReLU()

        self.gcn_m1_1 = GCNConv(args.hidden, args.hidden)
        self.gcn_m1_2 = GCNConv(args.hidden, args.hidden)

        self.gcn_d1_1 = GCNConv(args.hidden, args.hidden)
        self.gcn_d1_2 = GCNConv(args.hidden, args.hidden)

    def forward(self, args, data):
        # region  特征提取

        miRNA_number = len(data['mm_f']['Data_M'])
        disease_number = len(data['dd_s']['Data_M'])
        # 处理mm_f
        mm_f = torch.randn(miRNA_number, args.hidden, device=args.device)
        mm_f_1 = self.gcn_m1_1(mm_f, data['mm_f']['edges'],
                               data['m_s']['Data_M'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]])
        mm_f_1 = self.ReLU(mm_f_1)

        mm_f_2 = self.gcn_m1_2(mm_f_1, data['mm_f']['edges'],
                               data['m_s']['Data_M'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]])

        # 处理dd_s
        d_s = torch.randn(disease_number, args.hidden, device=args.device)
        dd_s_1 = self.gcn_d1_1(d_s, data['dd_s']['edges'],
                               data['d_s']['Data_M'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]])
        dd_s_1 = self.ReLU(dd_s_1)

        dd_s_2 = self.gcn_d1_2(dd_s_1, data['dd_s']['edges'],
                               data['d_s']['Data_M'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]])

        # endregion
        mf = self.ReLU(mm_f_2)
        df = self.ReLU(dd_s_2)
        x = torch.cat((mf, df), dim=0)

        return x


class GAT_LP(nn.Module):
    def __init__(self, args):
        super(GAT_LP, self).__init__()
        self.MyFeature = GCN_GAT(args)
        self.ELU = nn.ELU()

        self.gat1 = GATv2Conv(args.GATf, args.GATf // 2, args.GATh, concat=False)
        self.gat2 = GATv2Conv(args.GATf // 2, args.GATf // 4, args.GATh, concat=False)
        self.JK = JumpingKnowledge('cat')
        self.JKLin = nn.Linear(args.GATf + args.GATf // 2 + args.GATf // 4, args.hidden)

    def forward(self, args, data, edge_index):
        JKList = []
        x = self.MyFeature(args, data)
        JKList.append(x)
        x = self.gat1(x, edge_index)
        x = self.ELU(x)
        JKList.append(x)
        x = self.gat2(x, edge_index)
        JKList.append(x)
        x = self.JK(JKList)
        x = self.JKLin(x)  # epoch 1993 train_loss 0.28973439 val_auc 0.9531 test_auc 0.9432

        return x


class GraphormerModel(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(GraphormerModel, self).__init__()
        self.fc1 = nn.Linear(n_rna, args.hidden)
        self.fc2 = nn.Linear(n_dis, args.hidden)
        self.degree_encoder = DegreeEncoder(args.head, args.hidden)
        self.spatial_encoder = SpatialEncoder(max_dist=8, num_heads=args.head)
        self.graphormer_layer = GraphormerLayer(
            feat_size=args.hidden,
            hidden_size=2048,
            num_heads=args.head
        )

    def forward(self, data, graph):
        M_Feature = data[0]
        D_Feature = data[1]
        M_Feature = self.fc1(M_Feature)
        D_Feature = self.fc2(D_Feature)

        x = torch.cat([M_Feature, D_Feature], dim=0)
        degree_embedding = self.degree_encoder(graph)
        x += degree_embedding
        x = x.unsqueeze(0)

        spatial_embedding = self.spatial_encoder(graph)
        bias = spatial_embedding

        out = self.graphormer_layer(x, bias)
        out = out.squeeze(0)

        return out


class G_Module(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(G_Module, self).__init__()
        self.GAT_LP = GAT_LP(args)
        self.Pho = GraphormerModel(args, n_rna, n_dis)
        self.Feature_Weight = Feature_Weight(args, n_rna, n_dis)
        self.MLP = MLP(args)
        self.n_rna = n_rna
        self.n_dis = n_dis

    def encode(self, args, data, edge_index, g):
        G = self.GAT_LP(args, data, edge_index)

        FM1_data = [data['m_s']['Data_M'], data['d_s']['Data_M']]
        T = self.Pho(FM1_data, g)
        Fw = self.Feature_Weight(G, T)
        return Fw

    def decode(self, z, edge_label_index):
        # z所有节点的特征向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        res = (src * dst)
        res = self.MLP(res)
        return res

    def forward(self, args, data, edge_index, edge_label_index, g):
        z = self.encode(args, data, edge_index, g)
        res = self.decode(z, edge_label_index)
        return res

