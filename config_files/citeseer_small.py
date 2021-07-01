from config_files.base import BaseArgs

class Args(BaseArgs):
    def __init__(self):
        self.graph_type = 'citeseer_small'       ### Ego-small



        self.batch_size = 16 # 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 10 # 50
        self.test_total_size = 1000 # 1000

        ### training config
        self.batch_ratio = 32 * (
                    32 // self.batch_size)  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epoch_train_start = 0
        self.epochs = 302 # 3000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 4000 # 750 # 100
        self.epochs_test = 4000 # 750 # 100
        self.epochs_log = 50
        self.epochs_save = 50
        self.training_portion = 0.8
        self.validation_portion = 0.2
        self.test_portion = 0.2

        ### Transformer settings
        self.d_model = 100
        self.d_word_vec = 100   ## should be equal to self.d_model
        self.d_inner_hid = 200
        self.d_k = 100
        self.d_v = 100


        ## optimizer:
        self.lr_list = [0.0002, 0.00007, 0.00001]
        self.milestones = [self.batch_ratio * 100, self.batch_ratio * 200]


        self.note = 'Gransformer-3layers-nomodellayernorm-estnumnodes' # -gattk4log' # -trainpr0.2,valpr0.2,testpr0.2 # gattk16batchnorm-grposenck4batchnorm-bfsincpar'

        super().__init__(self.graph_type, self.note)