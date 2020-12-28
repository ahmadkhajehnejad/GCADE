import torch
import torch.nn as nn

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

        if exclude_last:
            self.weight_nodes = nn.Parameter(torch.Tensor(num_agg_features_nodes, num_output_features_nodes), requires_grad=True).to(device)
            self.weight_edges = nn.Parameter(torch.Tensor(num_agg_features_edges, num_output_features_edges), requires_grad=True).to(device)
        else:
            self.weight_nodes = nn.Parameter(torch.Tensor(num_agg_features_nodes + num_input_features_nodes,
                                                          num_output_features_nodes), requires_grad=True).to(device)
            self.weight_edges = nn.Parameter(torch.Tensor(num_agg_features_edges + num_input_features_edges,
                                                          num_output_features_edges), requires_grad=True).to(device)
        self.bias_nodes = nn.Parameter(torch.Tensor(num_output_features_nodes), requires_grad=True).to(device)
        self.bias_edges = nn.Parameter(torch.Tensor(num_output_features_edges), requires_grad=True).to(device)

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
            def aggregate_for_node(x_node_1, x_edge):
                return torch.nn.ReLU()(
                    self.lin_agg_node_2( self.lin_agg_node_1( torch.cat( [x_node_1, x_edge], dim=1))))
        else:
            self.lin_agg_node_1 = nn.Linear(2 * self.num_input_features_nodes + self.num_input_features_edges,
                                            2 * self.num_agg_features_nodes).to(device)
            self.lin_agg_node_2 = nn.Linear(2 * self.num_agg_features_nodes, self.num_agg_features_nodes).to(device)
            def aggregate_for_node(x_node_1, x_node_2, x_edge):
                return torch.nn.ReLU()(
                    self.lin_agg_node_2( self.lin_agg_node_1( torch.cat( [x_node_1, x_node_2, x_edge], dim=1))))
        self.agg_node = aggregate_for_node

        self.lin_agg_edge_1 = nn.Linear(self.num_input_features_nodes + self.num_input_features_edges,
                                        2 * self.num_agg_features_edges).to(device)
        self.lin_agg_edge_2 = nn.Linear(2 * self.num_agg_features_edges, self.num_agg_features_edges).to(device)
        def aggregate_for_edge(x_node, x_edge):
            return torch.nn.ReLU()(
                self.lin_agg_edge_2( self.lin_agg_edge_1( torch.cat( [x_node, x_edge], dim=1))))
        self.agg_edge = aggregate_for_edge




    def _update_edge(self, input_nodes, input_edges):
        batch_size = input_nodes.size(0)
        if input_nodes.size(1) == 0:
            agg_precedings = torch.zeros(batch_size, self.num_agg_features_edges, dtype=torch.float).to(self.device)
        else:
            list_precedings = [self.agg_edge(input_nodes[:,i,:], input_edges[:,i,:]).unsqueeze(1)
                               for i in range(input_nodes.size(1))]
            agg_precedings = torch.cat(list_precedings, dim=1).mean(dim=1)
        if self.exclude_last:
            concatenated = agg_precedings
        else:
            concatenated = torch.cat( [agg_precedings, input_edges[:,-1,:]], axis=1)
        multiplied = torch.matmul(concatenated, self.weight_edges)
        biased = multiplied + self.bias_edges.unsqueeze(0).repeat(batch_size, 1)
        activated = self.activation_edges(biased)
        return activated


    def _update_node(self, input_nodes, input_edges):
        batch_size = input_nodes.size(0)
        if input_nodes.size(1) == 1:
            agg_precedings = torch.zeros(batch_size, self.num_agg_features_nodes, dtype=torch.float).to(self.device)
        else:
            list_precedings = [self.agg_node(input_nodes[:,i,:], input_nodes[:,-1,:], input_edges[:,i,:]).unsqueeze(1)
                               for i in range(input_nodes.size(1) - 1)]
            agg_precedings = torch.cat(list_precedings, dim=1).mean(dim=1)
        if self.exclude_last:
            concatenated = agg_precedings
        else:
            concatenated = torch.cat( [agg_precedings, input_nodes[:,-1,:]], axis=1)
        multiplied = torch.matmul(concatenated, self.weight_nodes)
        biased = multiplied + self.bias_edges.unsqueeze(0).repeat(batch_size, 1)
        activated = self.activation_edges(biased)
        return activated


    def forward(self, input_nodes, input_edges):

        list_output_nodes = []
        list_output_edges = []

        k = 0
        for i in range(self.n):
            k0 = k
            j0 = max(0, i - self.m)
            for j in range(j0, i):
                list_output_edges.append(
                    self._update_edge(input_nodes[:, j0:j, :], input_edges[:, k0:k + 1, :]).unsqueeze(1))
                k += 1

            list_output_nodes.append(
                self._update_node(input_nodes[:, j0:i+1, :], input_edges[:, k0:k, :]).unsqueeze(1))

        output_nodes = torch.cat(list_output_nodes, dim=1)
        output_edges = torch.cat(list_output_edges, dim=1)
        return output_nodes, output_edges


    def extra_repr(self):
        return '<AutoRegressiveGraphConvLayer, exclude_last={}\n' \
               'num_input_features_nodes={}, num_agg_features_node={}, num_output_features_nodes={}>\n' \
               'num_input_features_edges={}, num_agg_features_edge={}, num_output_features_edges={}>\n'.format(
            self.exclude_last,
            self.num_input_features_nodes, self.num_agg_features_nodes, self.num_output_features_nodes,
            self.num_input_features_edges, self.num_agg_features_edges, self.num_output_features_edges
        )
