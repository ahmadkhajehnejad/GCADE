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

        # if exclude_last:
        #     self.weight_nodes = nn.Parameter(torch.Tensor(num_agg_features_nodes, num_output_features_nodes), requires_grad=True).to(device)
        #     self.weight_edges = nn.Parameter(torch.Tensor(num_agg_features_edges, num_output_features_edges), requires_grad=True).to(device)
        # else:
        #     self.weight_nodes = nn.Parameter(torch.Tensor(num_agg_features_nodes + num_input_features_nodes,
        #                                                   num_output_features_nodes), requires_grad=True).to(device)
        #     self.weight_edges = nn.Parameter(torch.Tensor(num_agg_features_edges + num_input_features_edges,
        #                                                   num_output_features_edges), requires_grad=True).to(device)
        # self.bias_nodes = nn.Parameter(torch.Tensor(num_output_features_nodes), requires_grad=True).to(device)
        # self.bias_edges = nn.Parameter(torch.Tensor(num_output_features_edges), requires_grad=True).to(device)

        self.weight_nodes.data.uniform_(-0.1, 0.1)
        self.bias_nodes.data.uniform_(-0.1, 0.1)
        self.weight_edges.data.uniform_(-0.1, 0.1)
        self.bias_edges.data.uniform_(-0.1, 0.1)

        self.activation_nodes = activation_nodes
        self.activation_edges = activation_edges

        if self.exclude_last:
            self.lin_agg_node_1 = nn.Linear(self.num_input_features_nodes + self.num_input_features_edges,
                                            2 * self.num_agg_features_nodes).to(device)
            self.lin_agg_node_2 = nn.Linear(2 * self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            self.last_lin_nodes = nn.Linear(self.num_agg_features_nodes, self.num_output_features_nodes).to(device)
            # def aggregate_for_node(x_node_1, x_edge):
            #     return torch.nn.ReLU()(
            #         self.lin_agg_node_2( torch.nn.ReLU()(
            #             self.lin_agg_node_1( torch.cat( [x_node_1, x_edge], dim=1)))))
        else:
            self.lin_agg_node_1 = nn.Linear(2 * self.num_input_features_nodes + self.num_input_features_edges,
                                            2 * self.num_agg_features_nodes).to(device)
            self.lin_agg_node_2 = nn.Linear(2 * self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            self.last_lin_nodes = nn.Linear(self.num_agg_features_nodes + self.num_input_features_nodes,
                                            self.num_output_features_nodes).to(device)
            # def aggregate_for_node(x_node_1, x_node_2, x_edge):
            #     return torch.nn.ReLU()(
            #         self.lin_agg_node_2( torch.nn.ReLU()(
            #             self.lin_agg_node_1( torch.cat( [x_node_1, x_node_2, x_edge], dim=1)))))
        # self.agg_node = aggregate_for_node

        self.lin_agg_edge_1 = nn.Linear(self.num_input_features_nodes + self.num_input_features_edges,
                                        2 * self.num_agg_features_edges).to(device)
        self.lin_agg_edge_2 = nn.Linear(2 * self.num_agg_features_edges, self.num_agg_features_edges).to(device)
        self.lst_lin_edges = nn.Linear(self.num_agg_features_edges, self.num_output_features_edges).to(device)
        # def aggregate_for_edge(x_node, x_edge):
        #     return torch.nn.ReLU()(
        #         self.lin_agg_edge_2( torch.nn.ReLU()(
        #             self.lin_agg_edge_1( torch.cat( [x_node, x_edge], dim=1)))))
        # self.agg_edge = aggregate_for_edge

        node_idx_node_edge_pairs = []
        edge_idx_node_edge_pairs = []
        k = 0
        for i in range(1, self.n):
            j0 = max(0, i - self.m)
            node_idx_node_edge_pairs += list(range(j0,i))
            edge_idx_node_edge_pairs += list(range(k, k+i-j0))
            k += i - j0
        self.node_idx_node_edge_pairs = torch.from_numpy(np.array(node_idx_node_edge_pairs, dtype=np.int32)).to(device)
        self.node_idx_node_edge_pairs.requires_grad = False
        self.edge_idx_node_edge_pairs = torch.from_numpy(np.array(edge_idx_node_edge_pairs, dtype=np.int32)).to(device)
        self.edge_idx_node_edge_pairs.requires_grad = False

        self.prev_nodes_idx = torch.zeros(self.n * self.m, requires_grad=False, dtype=torch.int32).to(device)
        k = 0
        for i in range(self.n):
            t = min(i, self.m)
            self.prev_nodes_idx[(i+1) * self.m - t : (i+1) * self.m] = torch.tensor(list(range(k, k+t))) + 1
            k += t


        self.agg_normalization_node = (1 / self.m) * torch.ones(self.n, requires_grad=False).to(device)
        for i in range(1, self.m):
            self.agg_normalization_node[i] = 1 / i


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

        self.node1_idx_node_edge_node_triples = torch.from_numpy(np.array(node1_idx_node_edge_nodes_triples, dtype=np.int32)).to(device)
        self.node1_idx_node_edge_node_triples.requires_grad = False
        self.edge_idx_node_edge_node_triples = torch.from_numpy(np.array(edge_idx_node_edge_nodes_triples, dtype=np.int32)).to(device)
        self.edge_idx_node_edge_node_triples.requires_grad = False
        self.node2_idx_node_edge_node_triples = torch.from_numpy(np.array(node2_idx_node_edge_nodes_triples, dtype=np.int32)).to(device)
        self.node2_idx_node_edge_node_triples.requires_grad = False

        self.prev_edges_idx = torch.zeros(self.n * self.m * self.m, requires_grad=False, dtype=torch.int32).to(device)
        k = 0
        e = 0
        for i in range(self.n):
            j0 = max(0, i - self.m)
            for j in range(j0, i):
                self.prev_edges_idx[(e+1) * self.m - (j-j0) : (i+1) * self.m] = torch.tensor(list(range(k, k+j-j0))) + 1
                e += 1
                k += (j-j0)

        self.n_e = 0
        for i in range(n):
            self.n_e += min(i, self.m)
        self.agg_normalization_edge = torch.ones(self.n_e, requires_grad=False).to(device)
        k = 0
        for i in range(1, self.n):
            t = min(i, self.m)
            for j in range(1, t):
                self.agg_normalization[k] = 1 / j
                k += 1



    # def _update_edge(self, input_nodes, input_edges):
    #     batch_size = input_nodes.size(0)
    #     if input_nodes.size(1) == 0:
    #         agg_precedings = torch.zeros(batch_size, self.num_agg_features_edges, dtype=torch.float).to(self.device)
    #     else:
    #         list_precedings = [self.agg_edge(input_nodes[:,i,:], input_edges[:,i,:]).unsqueeze(1)
    #                            for i in range(input_nodes.size(1))]
    #         agg_precedings = torch.cat(list_precedings, dim=1).mean(dim=1)
    #     if self.exclude_last:
    #         concatenated = agg_precedings
    #     else:
    #         concatenated = torch.cat( [agg_precedings, input_edges[:,-1,:]], axis=1)
    #     multiplied = torch.matmul(concatenated, self.weight_edges)
    #     biased = multiplied + self.bias_edges.unsqueeze(0).repeat(batch_size, 1)
    #     activated = self.activation_edges(biased)
    #     return activated
    #
    #
    # def _update_node(self, input_nodes, input_edges):
    #     batch_size = input_nodes.size(0)
    #     if input_nodes.size(1) == 1:
    #         agg_precedings = torch.zeros(batch_size, self.num_agg_features_nodes, dtype=torch.float).to(self.device)
    #     else:
    #         list_precedings = [self.agg_node(input_nodes[:,i,:], input_nodes[:,-1,:], input_edges[:,i,:]).unsqueeze(1)
    #                            for i in range(input_nodes.size(1) - 1)]
    #         agg_precedings = torch.cat(list_precedings, dim=1).mean(dim=1)
    #     if self.exclude_last:
    #         concatenated = agg_precedings
    #     else:
    #         concatenated = torch.cat( [agg_precedings, input_nodes[:,-1,:]], axis=1)
    #     multiplied = torch.matmul(concatenated, self.weight_nodes)
    #     biased = multiplied + self.bias_edges.unsqueeze(0).repeat(batch_size, 1)
    #     activated = self.activation_edges(biased)
    #     return activated
    #
    #
    # def forward(self, input_nodes, input_edges):
    #
    #     list_output_nodes = []
    #     list_output_edges = []
    #
    #
    #
    #     k = 0
    #     for i in range(self.n):
    #         k0 = k
    #         j0 = max(0, i - self.m)
    #         for j in range(j0, i):
    #             list_output_edges.append(
    #                 self._update_edge(input_nodes[:, j0:j, :], input_edges[:, k0:k + 1, :]).unsqueeze(1))
    #             k += 1
    #
    #         list_output_nodes.append(
    #             self._update_node(input_nodes[:, j0:i+1, :], input_edges[:, k0:k, :]).unsqueeze(1))
    #
    #     output_nodes = torch.cat(list_output_nodes, dim=1)
    #     output_edges = torch.cat(list_output_edges, dim=1)
    #     return output_nodes, output_edges


    def forward(self, input_nodes, input_edges):

        batch_size = input_nodes.size(0)

        node_edge_pairs = torch.cat( [ input_nodes[:, self.node_idx_node_edge_pairs, :],
                                       input_edges[:, self.edge_idx_node_edge_pairs,:] ], dim=2)

        node_edge_pairs = node_edge_pairs.view(-1, node_edge_pairs.size(2))
        node_edge_aggs = torch.nn.ReLU()(self.lin_agg_node_2(torch.nn.ReLU()(self.lin_agg_node_1(node_edge_pairs))))
        node_edge_aggs = node_edge_aggs.view([batch_size, -1, node_edge_aggs.size(1)])
        node_edge_aggs = torch.cat([torch.zeros(batch_size, 1, node_edge_aggs.size(2)), node_edge_aggs], dim=1)

        prev_nodes_aggs = node_edge_aggs[:, self.prev_nodes_idx, :].view([batch_size, self.n, self.m, -1])
        prev_nodes_aggs = prev_nodes_aggs.sum(dim=2) * torch.tile(self.agg_normalization_node.view(1, -1, 1),
                                                                  [batch_size, 1, prev_nodes_aggs.size(2)])

        if self.exclude_last:
            nodes_net_input = prev_nodes_aggs
        else:
            nodes_net_input = torch.cat([prev_nodes_aggs, input_nodes], dim=2)
        nodes_net_input = nodes_net_input.view([-1, nodes_net_input.size(2)])

        output_nodes = self.activation_nodes(self.last_lin_nodes(nodes_net_input)).view([batch_size, self.n, -1])

        node_edge_node_triples = torch.cat([input_nodes[:, self.node1_idx_node_edge_nodes_triples, :],
                                            input_edges[:, self.edge_idx_node_edge_node_triples, :],
                                            input_nodes[:, self.node2_idx_node_edge_nodes_triples, :]], dim=2)

        node_edge_node_triples = node_edge_node_triples.view([-1, node_edge_node_triples.size(2)])
        node_edge_node_aggs = torch.nn.ReLU()(self.lin_agg_edge_2(torch.nn.ReLU()(self.lin_agg_edge_1(node_edge_node_triples))))

        node_edge_node_aggs = node_edge_node_aggs.view([batch_size, -1, node_edge_node_aggs.size(1)])
        node_edge_node_aggs = torch.cat([torch.zeros(batch_size, 1, node_edge_node_aggs.size(2)), node_edge_node_aggs], dim=1)

        prev_edges_aggs = node_edge_node_aggs[:, self.prev_edges_idx, :].view([batch_size, -1, self.m, node_edge_node_aggs.size(2)])
        prev_edges_aggs = prev_edges_aggs.sum(dim=2) * torch.tile(self.agg_normalization_edge.view(1, -1, 1),
                                                                  [batch_size, 1, prev_nodes_aggs.size(2)])

        if self.exclude_last:
            edges_net_input = prev_edges_aggs
        else:
            edges_net_input = torch.cat([prev_edges_aggs, input_edges], dim=2)
        edges_net_input = edges_net_input.view([-1, edges_net_input.size(2)])

        output_edges = self.activation_edges(self.last_lin_edges(edges_net_input)).view([batch_size, self.n_e, -1])

        return output_nodes, output_edges



def extra_repr(self):
        return '<AutoRegressiveGraphConvLayer, exclude_last={}\n' \
               'num_input_features_nodes={}, num_agg_features_node={}, num_output_features_nodes={}>\n' \
               'num_input_features_edges={}, num_agg_features_edge={}, num_output_features_edges={}>\n'.format(
            self.exclude_last,
            self.num_input_features_nodes, self.num_agg_features_nodes, self.num_output_features_nodes,
            self.num_input_features_edges, self.num_agg_features_edges, self.num_output_features_edges
        )
