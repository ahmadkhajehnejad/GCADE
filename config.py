import torch.nn as nn
import torch

class Args():
    def __init__(self):

        ### Which dataset is used to train the model
        self.graph_type = 'DD'    ### protein
        # self.graph_type = 'caveman'  ### Community ??
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
        # self.graph_type = 'citeseer'             ### Ego
        # self.graph_type = 'citeseer_small'       ### Ego-small

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        self.use_pre_saved_graphs = True # False #

        # if none, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max previous node that looks back
        self.max_seq_len = None

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input + 'model_save/'  # only for nll evaluation
        self.graph_save_path = self.dir_input + 'graphs/'
        self.figure_save_path = self.dir_input + 'figures/'
        self.timing_save_path = self.dir_input + 'timing/'
        self.figure_prediction_save_path = self.dir_input + 'figures_prediction/'
        self.nll_save_path = self.dir_input + 'nll/'


        self.batch_size = 32 # 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 50
        self.test_total_size = 1000 # 1000

        ### training config
        self.num_workers = 4  # num workers to load data, default 4
        self.batch_ratio = 32 * (32 // self.batch_size) # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epoch_train_start = 0
        self.epochs = 3002 # 3000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 750 # 100
        self.epochs_test = 750 # 100
        self.epochs_log = 100
        self.epochs_save = 50

        # self.lr = 0.003 #0.003
        # self.milestones = [4000, 10000]
        # self.lr_rate = 0.3

        self.sample_time = 2  # sample time in each time step, when validating

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('self.device:', self.device)


        self.node_ordering = 'bfs' # 'blanket' #
        self.use_max_prev_node = False # True #   ### ignored when using an input_type other than preceding_neighbors_vector'
        if self.use_max_prev_node:
            assert self.node_ordering in ['bfs']

        self.use_bfs_incremental_parent_idx = False # True #    ### now just implemented for max_pre_node_neighbors_vec input_type

        self.max_num_generate_trials = 1000

        self.input_type = 'preceding_neighbors_vector' # 'max_prev_node_neighbors_vec' # 'node_based' #
        self.only_encoder = True # False
        self.output_positional_embedding = False # True #   
        self.input_bfs_depth = False # True #  
        if self.input_bfs_depth:
            assert self.node_ordering in ['bfs']
        self.k_graph_attention = 0 # 4
        self.normalize_graph_attention = False # True # 
        if self.input_type == 'node_based':
            self.trg_pad_idx = 0
            self.src_pad_idx = 0
        elif self.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            self.trg_pad_idx = -2
            self.src_pad_idx = -2  # must be equal to self.trg_pad_idx
            # self.avg = False # True #
            # if self.avg == True:
            #     self.zero_input = 0
            #     self.one_input = 1
            #     self.dontcare_input = 0
            # else:
            if True:
                self.zero_input = -1
                self.one_input = 1
                self.dontcare_input = 0
        else:
            raise NotImplementedError


        ### Transformer settings

        self.use_MADE = False # True #  
        self.MADE_num_masks = 3 # 1
        self.MADE_natural_ordering = False # True #
        self.MADE_num_hidden_layers = 1 # 3
        self.d_model = 4 * 100 # 512
        self.d_word_vec = 4 * 100 # 512   ## should be equal to self.d_model
        self.d_inner_hid = 4 * 400 # 2048
        self.d_k = 2 * 4 * 50 # 64
        self.d_v = 2 * 4 * 50 # 64
        self.n_layers = 2 # 6
        self.n_grlayers = 0
        assert self.n_grlayers <= self.n_layers
        assert self.n_grlayers == 0 or self.k_graph_attention == 0
        self.n_head = 1 # 8
        self.ensemble_input_type = 'repeat' # 'multihop-single' # 'negative' # 'multihop' #
        if self.ensemble_input_type == 'negative':
            self.n_ensemble = 2
        elif self.ensemble_input_type == 'multihop':
            self.ensemble_multihop = [2]
            self.n_ensemble = len(self.ensemble_multihop) + 1
        elif self.ensemble_input_type == 'multihop-single':
            self.ensemble_multihop = [1,2,3,4]
            self.n_ensemble = 1
        elif self.ensemble_input_type == 'repeat':
            self.n_ensemble = 1 # 8
        else:
            raise NotImplementedError('ensemble_input_type', self.ensemble_input_type, 'not recognized.')
        self.dropout = 0.1
        self.proj_share_weight = False # True
        self.embs_share_weight = True
        self.scale_emb_or_prj = 'prj'

        ## optimizer:
        # self.epochs = 200
        self.lr_mul = 2.0
        self.n_warmup_steps = 500 # 4000

        ### output
        self.use_tb = False  # use tensorboard
        self.output_dir = './output'

        self.note = 'Gransformer-2layers-bfsincpar'
        if self.note == 'Gransformer-3layers':
            self.n_layers = 3
            self.n_grlayers = 0
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-2layers':
            self.n_layers = 2
            self.n_grlayers = 0
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-2layers-bfsincpar':
            self.n_layers = 2
            self.n_grlayers = 0
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = True
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-4layers-2grlayers':
            self.n_layers = 4
            self.n_grlayers = 2
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-3layers-1grlayer':
            self.n_layers = 3
            self.n_grlayers = 1
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-2layers-1grlayer':
            self.n_layers = 2
            self.n_grlayers = 1
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
        elif self.note == 'Gransformer-gattk4':
            self.n_layers = 2
            self.n_grlayers = 0
            self.node_ordering = 'bfs'
            self.use_max_prev_node = False
            self.use_bfs_incremental_parent_idx = False
            self.k_graph_attention = 0
            self.n_ensemble = 1
            self.n_head = 1
            self.k_graph_attention = 4

        ### filenames to save intemediate and final outputs
        # self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        self.fname = self.note
        #     self.hidden_size_rnn) + '_'
        # self.fname_pred = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_pred_'
        self.fname_pred = self.note + '_' + self.graph_type + '_' + self.input_type +  '_pred_'
        # self.fname_train = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_train_'
        self.fname_train = self.note + '_' + self.graph_type + '_' + self.input_type + '_train_'
        # self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(
        #     self.hidden_size_rnn) + '_test_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + self.input_type + '_test_'
        # self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline + '_' + self.metric_baseline
