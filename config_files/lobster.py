from config_files.base import BaseArgs

class Args(BaseArgs):
    def __init__(self):
        self.graph_type = 'lobster'       ###



        self.batch_size = 16 # 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 15 # 50
        self.test_total_size = 1000 # 1000

        ### training config
        self.batch_ratio = 32 * (
                    32 // self.batch_size)  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epoch_train_start = 0 # 0
        self.epochs = 3002 # 3002  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 4000 # 750 # 100
        self.epochs_test = 4000 # 750 # 100
        self.epochs_log = 50
        self.epochs_save = 50
        self.training_portion = 0.8
        self.validation_portion = 0.2
        self.test_portion = 0.2

        ### Transformer settings
        self.d_model = 200
        self.d_word_vec = 200   ## should be equal to self.d_model
        self.d_inner_hid = 400
        self.d_k = 200
        self.d_v = 200

        ## optimizer:
        self.lr_list = [0.0001] # [0.0005, 0.0001, 0.00002, 0.005]  # , 0.00002, 0.000004] # [0.0001, 0.00002, 0.000004, 0.0000008, 0.00000016]
        self.milestones = [] #  [self.batch_ratio * 100, self.batch_ratio * 200, self.batch_ratio * 800]  #[self.batch_ratio * 250, self.batch_ratio * 750] # [self.batch_ratio * 60, self.batch_ratio * 120, self.batch_ratio * 180, self.batch_ratio * 240]

        self.note = 'Gransformer-6layers-nomodellayernorm-feedgraphlength-MADEhl2msk1natuord1dimred1-typededges-gattk16norm-grposenck16norm' # -noaddposenc' # -estnumnodes-exactgen' # -bfsincpar' # ' #-sepoptepoch20 # -separateTerminationBit' # '  # -allowAllZeros' #' #   ' # -useminnumnodes' #  ' # # ' # ' # -weightpositions' # ' # -posoutputoneHot' # -posoutputtril' #  -nhead1-nensemble1' # ' # -trainpr0.2,valpr0.2,testpr0.2 #'

        super().__init__(self.graph_type, self.note, self.batch_ratio)
