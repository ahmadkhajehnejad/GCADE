import torch.nn as nn

n = 100
m = 20
list_layer_size = [
    {
        'input_features_nodes': n+1, 'agg_features_nodes': 100, 'output_features_nodes':100,
        'input_features_edges': 1, 'agg_features_edges': 100, 'output_features_edges':100,
        'activation_nodes': nn.ReLU(), 'activation_edges': nn.ReLU()
    },
    {
        'input_features_nodes': 100, 'agg_features_nodes': 100, 'output_features_nodes':50,
        'input_features_edges': 100, 'agg_features_edges': 100, 'output_features_edges':50,
        'activation_nodes': nn.ReLU(), 'activation_edges': nn.ReLU()
    },
    {
        'input_features_nodes': 50, 'agg_features_nodes': 50, 'output_features_nodes':1,
        'input_features_edges': 50, 'agg_features_edges': 50, 'output_features_edges':1,
        'activation_nodes': nn.Sigmoid(), 'activation_edges': nn.Sigmoid()
    },
]