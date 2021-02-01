import torch
import torch.nn as nn
import numpy as np

class AutoRegressiveGraphConvLayer(nn.Module):

    def __init__(self, n, m, num_input_features_nodes, num_input_features_edges,
                 num_agg_features_nodes, num_agg_features_edges, num_output_features_nodes, num_output_features_edges,
                 activation_nodes, activation_edges, exclude_last=False, device='cpu'):

        super(AutoRegressiveGraphConvLayer, self).__init__()

        self.device = device

        self.n = n
        self.m = m

        self.num_input_features_nodes = num_input_features_nodes
        self.num_input_features_edges = num_input_features_edges
        self.num_agg_features_nodes = num_agg_features_nodes
        self.num_agg_features_edges = num_agg_features_edges
        self.num_output_features_nodes = num_output_features_nodes
        self.num_output_features_edges = num_output_features_edges
        self.exclude_last = exclude_last

        self.activation_nodes = activation_nodes
        self.activation_edges = activation_edges

        # print('\n\n--------')
        # print(self.n, self.m)
        # print(self.num_input_features_nodes, self.num_input_features_edges)
        # print(self.num_agg_features_nodes, self.num_agg_features_edges)
        # print(self.num_output_features_nodes, self.num_output_features_edges)
        # print(self.exclude_last)
        # print(self.activation_nodes, self.activation_edges)
        # print('________\n\n')


        ##### making node-edge pair indices

        node_idx_node_edge_pairs = []
        edge_idx_node_edge_pairs = []
        k = 0
        for i in range(1, self.n):
            j0 = max(0, i - self.m)
            node_idx_node_edge_pairs += list(range(j0,i))
            edge_idx_node_edge_pairs += list(range(k, k+i-j0))
            k += i - j0
        self.node_idx_node_edge_pairs = torch.from_numpy(np.array(node_idx_node_edge_pairs, dtype=np.long)).to(device)
        self.node_idx_node_edge_pairs.requires_grad = False
        self.edge_idx_node_edge_pairs = torch.from_numpy(np.array(edge_idx_node_edge_pairs, dtype=np.long)).to(device)
        self.edge_idx_node_edge_pairs.requires_grad = False

        ####### making node-edge-node triple indices

        node1_idx_node_edge_nodes_triples = []
        edge_idx_node_edge_nodes_triples = []
        node2_idx_node_edge_nodes_triples = []
        k = 0
        for i in range(1, self.n):
            j0 = max(0, i - self.m)
            node1_idx_node_edge_nodes_triples += list(range(j0, i))
            edge_idx_node_edge_nodes_triples += list(range(k, k + i - j0))
            node2_idx_node_edge_nodes_triples += [i for _ in range(j0, i)]
            k += i - j0

        self.node1_idx_node_edge_node_triples = torch.from_numpy(
            np.array(node1_idx_node_edge_nodes_triples, dtype=np.long)).to(device)
        self.node1_idx_node_edge_node_triples.requires_grad = False
        self.edge_idx_node_edge_node_triples = torch.from_numpy(
            np.array(edge_idx_node_edge_nodes_triples, dtype=np.long)).to(device)
        self.edge_idx_node_edge_node_triples.requires_grad = False
        self.node2_idx_node_edge_node_triples = torch.from_numpy(
            np.array(node2_idx_node_edge_nodes_triples, dtype=np.long)).to(device)
        self.node2_idx_node_edge_node_triples.requires_grad = False

        #######   linear layers for updating nodes

        if self.exclude_last:
            self.lin_agg_node_1 = nn.Linear(self.num_input_features_nodes + self.num_input_features_edges,
                                            2 * self.num_agg_features_nodes).to(device)
            self.lin_agg_node_2 = nn.Linear(2 * self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            self.last_lin_nodes_1 = nn.Linear(self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            self.last_lin_nodes_2 = nn.Linear(self.num_agg_features_nodes, self.num_output_features_nodes).to(device)
        else:
            self.lin_agg_node_1 = nn.Linear(2 * self.num_input_features_nodes + self.num_input_features_edges,
                                            2 * self.num_agg_features_nodes).to(device)
            self.lin_agg_node_2 = nn.Linear(2 * self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            self.last_lin_nodes_1 = nn.Linear(self.num_agg_features_nodes + self.num_input_features_nodes,
                                            self.num_agg_features_nodes + self.num_input_features_nodes).to(device)
            self.last_lin_nodes_2 = nn.Linear(self.num_agg_features_nodes + self.num_input_features_nodes,
                                              self.num_output_features_nodes).to(device)
        #### making indices for updating nodes


        self.prev_nodes_idx = torch.zeros(self.n * self.m, requires_grad=False, dtype=torch.long).to(device)
        k = 0
        for i in range(self.n):
            t = min(i, self.m)
            self.prev_nodes_idx[(i+1) * self.m - t : (i+1) * self.m] = torch.tensor(list(range(k, k+t))) + 1
            k += t

        self.agg_normalization_node = (1 / self.m) * torch.ones(self.n, requires_grad=False).to(device)
        for i in range(1, self.m):
            self.agg_normalization_node[i] = 1 / i

        ### linear layers for updating edges

        self.lin_agg_edge_1 = nn.Linear(self.num_input_features_nodes + self.num_input_features_edges,
                                        2 * self.num_agg_features_edges).to(device)
        self.lin_agg_edge_2 = nn.Linear(2 * self.num_agg_features_edges, self.num_agg_features_edges).to(device)
        if self.exclude_last:
            self.last_lin_edges_1 = nn.Linear(self.num_agg_features_edges, self.num_agg_features_edges).to(device)
            self.last_lin_edges_2 = nn.Linear(self.num_agg_features_edges, self.num_output_features_edges).to(device)
        else:
            self.last_lin_edges_1 = nn.Linear(self.num_agg_features_edges + self.num_input_features_edges,
                                            self.num_agg_features_edges + self.num_input_features_edges).to(device)
            self.last_lin_edges_2 = nn.Linear(self.num_agg_features_edges + self.num_input_features_edges,
                                            self.num_output_features_edges).to(device)


        #### making indices for updating edges

        self.n_e = 0
        for i in range(n):
            self.n_e += min(i, self.m)

        self.prev_edges_idx = torch.zeros(self.n_e * self.m, requires_grad=False, dtype=torch.long).to(device)
        k = 0
        e = 0
        for i in range(self.n):
            j0 = max(0, i - self.m)
            for j in range(j0, i):
                if j > j0:
                    self.prev_edges_idx[(e+1) * self.m - (j-j0) : (e+1) * self.m] = torch.tensor(list(range(k, k+j-j0))) + 1
                e += 1
            k += (i-j0)


        self.agg_normalization_edge = torch.ones(self.n_e, requires_grad=False).to(device)
        k = 0
        for i in range(1, self.n):
            t = min(i, self.m)
            for j in range(t):
                if j > 0:
                    self.agg_normalization_edge[k] = 1 / j
                k += 1


    def forward(self, input_nodes, input_edges):

        batch_size = input_nodes.size(0)


        node_edge_pairs = torch.cat([input_nodes[:, self.node_idx_node_edge_pairs, :],
                                     input_edges[:, self.edge_idx_node_edge_pairs, :]], dim=2)
        node_edge_pairs = node_edge_pairs.view(-1, node_edge_pairs.size(2))

        node_edge_node_triples = torch.cat([input_nodes[:, self.node1_idx_node_edge_node_triples, :],
                                            input_edges[:, self.edge_idx_node_edge_node_triples, :],
                                            input_nodes[:, self.node2_idx_node_edge_node_triples, :]], dim=2)
        node_edge_node_triples = node_edge_node_triples.view([-1, node_edge_node_triples.size(2)])


        ### Update nodes


        # print('\n\n-----------------', self.exclude_last, '------------------------\n\n')
        # print('\n\n-----------------', node_edge_node_triples.size(), '------------------------\n\n')
        # print('\n\n-----------------', self.lin_agg_node_1, '------------------------\n\n')

        if self.exclude_last:
            node_edge_aggs = torch.nn.ReLU()(self.lin_agg_node_2(torch.nn.ReLU()(self.lin_agg_node_1(node_edge_pairs))))
            node_edge_aggs = node_edge_aggs.view([batch_size, -1, node_edge_aggs.size(1)])
            node_edge_aggs = torch.cat([torch.zeros(batch_size, 1, node_edge_aggs.size(2)).to(self.device), node_edge_aggs], dim=1)
            tmp_aggs = node_edge_aggs
        else:
            node_edge_node_aggs = torch.nn.ReLU()(self.lin_agg_node_2(torch.nn.ReLU()(self.lin_agg_node_1(node_edge_node_triples))))
            node_edge_node_aggs = node_edge_node_aggs.view([batch_size, -1, node_edge_node_aggs.size(1)])
            node_edge_node_aggs = torch.cat([torch.zeros(batch_size, 1, node_edge_node_aggs.size(2)).to(self.device), node_edge_node_aggs], dim=1)
            tmp_aggs = node_edge_node_aggs
        prev_nodes_aggs = tmp_aggs[:, self.prev_nodes_idx, :].view([batch_size, self.n, self.m, -1])
        prev_nodes_aggs = prev_nodes_aggs.sum(dim=2) * self.agg_normalization_node.view(1, -1, 1).repeat(
                                                                        batch_size, 1, prev_nodes_aggs.size(3))

        if self.exclude_last:
            nodes_net_input = prev_nodes_aggs
        else:
            nodes_net_input = torch.cat([prev_nodes_aggs, input_nodes], dim=2)
        nodes_net_input = nodes_net_input.view([-1, nodes_net_input.size(2)])

        output_nodes = self.activation_nodes(self.last_lin_nodes_2(torch.nn.ReLU()(self.last_lin_nodes_1(nodes_net_input)))).view([batch_size, self.n, -1])

        ### Update edges

        node_edge_aggs = torch.nn.ReLU()(self.lin_agg_edge_2(torch.nn.ReLU()(self.lin_agg_edge_1(node_edge_pairs))))
        node_edge_aggs = node_edge_aggs.view([batch_size, -1, node_edge_aggs.size(1)])
        node_edge_aggs = torch.cat([torch.zeros(batch_size, 1, node_edge_aggs.size(2)).to(self.device), node_edge_aggs], dim=1)

        prev_edges_aggs = node_edge_aggs[:, self.prev_edges_idx, :].view([batch_size, -1, self.m, node_edge_aggs.size(2)])
        prev_edges_aggs = prev_edges_aggs.sum(dim=2) * self.agg_normalization_edge.view(1, -1, 1).repeat(
                                                                        batch_size, 1, prev_edges_aggs.size(3))

        if self.exclude_last:
            edges_net_input = prev_edges_aggs
        else:
            edges_net_input = torch.cat([prev_edges_aggs, input_edges], dim=2)
        edges_net_input = edges_net_input.view([-1, edges_net_input.size(2)])

        output_edges = self.activation_edges(self.last_lin_edges_2(torch.nn.ReLU()(self.last_lin_edges_1(edges_net_input)))).view([batch_size, self.n_e, -1])

        return output_nodes, output_edges



def extra_repr(self):
        return '<AutoRegressiveGraphConvLayer, exclude_last={}\n' \
               'num_input_features_nodes={}, num_agg_features_node={}, num_output_features_nodes={}>\n' \
               'num_input_features_edges={}, num_agg_features_edge={}, num_output_features_edges={}>\n'.format(
            self.exclude_last,
            self.num_input_features_nodes, self.num_agg_features_nodes, self.num_output_features_nodes,
            self.num_input_features_edges, self.num_agg_features_edges, self.num_output_features_edges
        )
