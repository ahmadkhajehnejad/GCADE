import torch.nn as nn
from layers import AutoRegressiveGraphConvLayer
import config

class GCADEModel(nn.Module):

    def __init__(self):
        super(GCADEModel, self).__init__()
        self.layers = [AutoRegressiveGraphConvLayer(config.n, config.m,
                                                    layer_size['input_features_nodes'], layer_size['input_features_edges'],
                                                    layer_size['agg_features_nodes'], layer_size['agg_features_edges'],
                                                    layer_size['output_features_nodes'], layer_size['output_features_edges'],
                                                    nn.ReLU(), nn.ReLU(), exclude_last = (i == len(config.list_layer_size) - 1)
                                                    )
                       for i, layer_size in enumerate(config.list_layer_size)]

        list_layer_size = config.list_layer_size

        for i, layer_size in enumerate(list_layer_size):
            setattr(self, 'lay_' + str(i), AutoRegressiveGraphConvLayer(config.n, config.m,
                                                                        list_layer_size[0]['input_features_nodes'],
                                                                        list_layer_size[0]['input_features_edges'],
                                                                        list_layer_size[0]['agg_features_nodes'],
                                                                        list_layer_size[0]['agg_features_edges'],
                                                                        list_layer_size[0]['output_features_nodes'],
                                                                        list_layer_size[0]['output_features_edges'],
                                                                        nn.ReLU(), nn.ReLU(),
                                                                        exclude_last=(i == len(list_layer_size) - 1)))

    def forward(self, input_nodes, input_edges):
        output_nodes, output_edges = input_nodes, input_edges
        for layer in self.layes:
            output_nodes, output_edges = layer(output_nodes, output_edges)
        return output_nodes, output_edges

'''
def train(dataset_train):

    epoch = 1

    # initialize optimizer
    optimizer = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)
'''