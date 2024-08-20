import torch
import os.path as osp
import libgptb.losses as L
import torch_geometric.transforms as T
import libgptb.augmentors as A
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from libgptb.evaluators import get_split, LREvaluator
from libgptb.models import SingleBranchContrast, DualBranchContrast
from torch_geometric.nn import global_add_pool,  GINConv, MessagePassing
from torch_geometric.nn.inits import uniform
from libgptb.model.abstract_gcl_model import AbstractGCLModel


class GINConvFirstLayer(MessagePassing):
    def __init__(self, input_dim, out_dim):
        super(GINConvFirstLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.mlp(aggr_out + x)

class GINConvAggregation(MessagePassing):
    def __init__(self):
        super(GINConvAggregation, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers-1:
                self.layers.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.layers.append(GINConvAggregation())
                self.batch_norms.append(nn.BatchNorm1d(input_dim))
            

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)

class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)

class InfoGraphGINRev(AbstractGCLModel):
    def __init__(self, config, data_feature):
        
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 2)
        self.device = config.get('device', torch.device('cpu'))
        self.loss_type = config.get('loss_type', 'JSD')
        self.input_dim = data_feature.get('input_dim', 2)
        super().__init__(config, data_feature)

        gconv = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, activation=torch.nn.ReLU, num_layers=self.layers).to(self.device)
        fc1 = FC(hidden_dim=self.input_dim*(self.layers-1)+self.nhid)
        fc2 = FC(hidden_dim=self.input_dim*(self.layers-1)+self.nhid)

        self.encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(self.device)
        if self.loss_type == 'JSD':
            self.contrast_model = SingleBranchContrast(loss=L.DebiasedJSD(), mode='G2L').to(self.device) # L.JSD() for JSD loss
        elif self.loss_type == 'InfoNCE':
            print('Using InfoNCE loss')
            self.contrast_model = SingleBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2L').to(self.device) # L.InfoNCE() for InfoNCE loss