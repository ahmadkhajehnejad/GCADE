import torch.nn as nn

class Args():
    def __init__(self):

        ### Which dataset is used to train the model
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)


        # if none, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max previous node that looks back

        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 32

        ### training config
        self.num_workers = 4  # num workers to load data, default 4
        self.batch_ratio = 32  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 3000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 100
        self.epochs_test = 100
        self.epochs_log = 100
        self.epochs_save = 100

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2  # sample time in each time step, when validating

    def list_layer_sizes(self):
        return [
            {
                'input_features_nodes': self.max_num_node + 1, 'agg_features_nodes': 100, 'output_features_nodes': 100,
                'input_features_edges': 1, 'agg_features_edges': 100, 'output_features_edges': 100,
                'activation_nodes': nn.ReLU(), 'activation_edges': nn.ReLU()
            },
            {
                'input_features_nodes': 100, 'agg_features_nodes': 100, 'output_features_nodes': 50,
                'input_features_edges': 100, 'agg_features_edges': 100, 'output_features_edges': 50,
                'activation_nodes': nn.ReLU(), 'activation_edges': nn.ReLU()
            },
            {
                'input_features_nodes': 50, 'agg_features_nodes': 50, 'output_features_nodes': 1,
                'input_features_edges': 50, 'agg_features_edges': 50, 'output_features_edges': 1,
                'activation_nodes': nn.Sigmoid(), 'activation_edges': nn.Sigmoid()
            },
        ]
