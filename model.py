import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from layers import AutoRegressiveGraphConvLayer
import numpy as np
import time

class GCADEModel(nn.Module):

    def __init__(self, args):
        super(GCADEModel, self).__init__()
        # self.layers = [AutoRegressiveGraphConvLayer(args.max_num_node, args.max_prev_node,
        #                                             layer_size['input_features_nodes'], layer_size['input_features_edges'],
        #                                             layer_size['agg_features_nodes'], layer_size['agg_features_edges'],
        #                                             layer_size['output_features_nodes'], layer_size['output_features_edges'],
        #                                             nn.ReLU(), nn.ReLU(), exclude_last = (i == len(config.list_layer_size) - 1)
        #                                             )
        #                for i, layer_size in enumerate(config.list_layer_size)]

        list_layer_sizes = args.list_layer_sizes()

        self.num_layers = len(list_layer_sizes)
        for i, layer_size in enumerate(list_layer_sizes):
            setattr(self, 'lay_' + str(i), AutoRegressiveGraphConvLayer(args.max_num_node, args.max_prev_node,
                                                                        layer_size['input_features_nodes'],
                                                                        layer_size['input_features_edges'],
                                                                        layer_size['agg_features_nodes'],
                                                                        layer_size['agg_features_edges'],
                                                                        layer_size['output_features_nodes'],
                                                                        layer_size['output_features_edges'],
                                                                        nn.ReLU(), nn.ReLU(),
                                                                        exclude_last=(i == len(list_layer_sizes) - 1)))

    def forward(self, input_nodes, input_edges):
        output_nodes, output_edges = input_nodes, input_edges
        for i in range(self.num_layers):
            layer = getattr(self, 'lay_' + str(i))
            output_nodes, output_edges = layer(output_nodes, output_edges)
        return output_nodes, output_edges

def nll(input_nodes, input_edges, output_nodes, output_edges, len_):
    max_n = input_nodes.size(1)
    batch_size = input_nodes.size(0)
    res = torch.zeros(batch_size)
    k = 0
    for i in range(max_n):
        ind = torch.gt(len_, i)
        tmp_1 = torch.log(output_nodes[ind, i, 0])
        tmp_2 = torch.log(output_edges[ind, k:k+i, 0]) * input_edges[ind, k:k+i, 0] + \
            torch.log(1 - output_edges[ind, k:k+i, 0]) * (1 - input_edges[ind, k:k+i, 0])
        res[ind] += tmp_1 + tmp_2
        k += i

        if i < max_n - 1:
            ind = torch.eq(len_, i+1)
            tmp = torch.log(1 - output_nodes[ind, i+1,0])
            res[ind] += tmp
    return res


def train(gcade_model, dataset_train, args):

    # initialize optimizer
    optimizer = torch.optim.Adam(list(gcade_model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        time_start = time.time()
        running_loss = 0.0
        trsz = 0
        gcade_model.train()
        for i, data in enumerate(dataset_train, 0):
            input_nodes = data['input_nodes_features'].float()
            input_edges = data['input_edges_features'].float()
            len_ = data['len'].float()

            optimizer.zero_grad()
            pred_nodes, pred_edges = gcade_model(input_nodes, input_edges)
            loss = nll(pred_nodes, pred_edges, input_nodes, input_edges, len_)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            trsz += input_nodes.size(0)

        print('[epoch %d]     loss: %.3f' %
              (epoch + 1, running_loss / trsz))
        time_end = time.time()
        time_all[epoch - 1] = time_end - time_start
    #     # test
    #     if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
    #         for sample_time in range(1,4):
    #             G_pred = []
    #             while len(G_pred)<args.test_total_size:
    #                 if 'GraphRNN_VAE' in args.note:
    #                     G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
    #                 elif 'GraphRNN_MLP' in args.note:
    #                     G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
    #                 elif 'GraphRNN_RNN' in args.note:
    #                     G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
    #                 G_pred.extend(G_pred_step)
    #             # save graphs
    #             fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
    #             save_graph_list(G_pred, fname)
    #             if 'GraphRNN_RNN' in args.note:
    #                 break
    #         print('test done, graphs saved')
    #
    #
    #     # save model checkpoint
    #     if args.save:
    #         if epoch % args.epochs_save == 0:
    #             fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
    #             torch.save(rnn.state_dict(), fname)
    #             fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
    #             torch.save(output.state_dict(), fname)
    #     epoch += 1
    # np.save(args.timing_save_path+args.fname,time_all)
