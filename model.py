import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from layers import AutoRegressiveGraphConvLayer
import numpy as np
import time
import sys
#from utils import get_graph, save_graph_list
import utils
# from data import my_decode_adj
import data


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
        for i, layer_info in enumerate(list_layer_sizes):
            setattr(self, 'lay_' + str(i), AutoRegressiveGraphConvLayer(args.max_num_node, args.max_prev_node,
                                                                        layer_info['input_features_nodes'],
                                                                        layer_info['input_features_edges'],
                                                                        layer_info['agg_features_nodes'],
                                                                        layer_info['agg_features_edges'],
                                                                        layer_info['output_features_nodes'],
                                                                        layer_info['output_features_edges'],
                                                                        layer_info['num_aggregation_layers_nodes'],
                                                                        layer_info['num_aggregation_layers_edges'],
                                                                        layer_info['num_last_linear_layers_nodes'],
                                                                        layer_info['num_last_linear_layers_edges'],
                                                                        layer_info['activation_nodes'],
                                                                        layer_info['activation_edges'],
                                                                        exclude_last=(i == len(list_layer_sizes) - 1),
                                                                        device=args.device))

    def forward(self, input_nodes, input_edges):
        output_nodes, output_edges = input_nodes, input_edges
        for i in range(self.num_layers):
            layer = getattr(self, 'lay_' + str(i))
            output_nodes, output_edges = layer(output_nodes, output_edges)
        return output_nodes, output_edges

def nll(output_nodes, output_edges, input_nodes, input_edges, len_, args, batch_num):

    # print(output_nodes.min().item(), output_nodes.max().item(), input_nodes.min().item(), input_edges.max().item())

    # output_nodes = output_nodes * 0.99 + 0.005
    # output_edges = output_edges * 0.99 + 0.005

    max_n = args.max_num_node
    batch_size = input_nodes.size(0)
    res = torch.zeros(batch_size).to(args.device)
    k = 0
    for i in range(max_n):
        ind = torch.gt(len_, i)
        # ind = torch.gt(len_, 0)

        # if i < 5:
        #     print(i, ind.sum().item(), output_nodes[ind, i, 0].mean().item())

        tmp_1 = torch.log(output_nodes[ind, i, 0])
        t = min(i, args.max_prev_node)
        tmp_2 = torch.log(output_edges[ind, k:k+t, 0]) * input_edges[ind, k:k+t, 0] + \
            torch.log(1 - output_edges[ind, k:k+t, 0]) * (1 - input_edges[ind, k:k+t, 0])
        # if i == 1 and batch_num == 31 and ind.sum().item() > 0:
        #     print('\n', i,
        #     list(zip(list(input_edges[ind, k:k+t, 0][0].detach().cpu().numpy()), list(output_edges[ind, k:k+t, 0][0].detach().cpu().numpy()), tmp_2[0].detach().cpu().numpy())),
                  # tmp_2[0].mean().item(),
                  # input_edges[ind, k:k+t, 0][0].mean().item(),
                  # output_edges[ind, k:k + t, 0][0].mean().item())
        # print('\n', i, input_edges[ind, k:k+t, 0][0].size(), output_edges[ind, k:k+t,0][0].detach().cpu().numpy().shape)
        # print('\n', i, ind.size(), input_edges.size(), input_edges[ind].size())
        # print(i, ind.sum().item(), input_edges[ind, k:k + t, 0].sum(dim=1).mean().item(),
        #                         output_edges[ind, k:k + t].sum(dim=1).mean().item())
        # print(i, ind.sum().item(), tmp_1.mean().item(), output_nodes[ind, i].mean().item())  # tmp_2.mean().item())
        p_ = 0
        res[ind] += (1 / (i+1)) ** p_ * (tmp_2.sum(dim=1) + tmp_1)
        k += t

        if i < max_n - 1:
            ind = torch.eq(len_, i+1)
            tmp = torch.log(1 - output_nodes[ind, i+1,0])
            res[ind] += (1 / (i+1)) ** p_ * tmp
    # if batch_num == 31:
        # print('\n')
        # input()
    return -res.sum()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def generate_graph(model, args):
    model.eval()

    first_layer = model.lay_0
    if args.feed_node_id:
        tmp = np.concatenate([np.zeros([args.max_num_node, 1], dtype=np.float32), np.eye(args.max_num_node, dtype=np.float32)], axis=1)
        input_nodes = torch.tensor(np.tile( np.expand_dims(tmp, axis=0), [args.test_batch_size, 1, 1])).to(args.device)
    else:
        input_nodes = torch.zeros(args.test_batch_size, args.max_num_node, 1).to(args.device)
    if args.feed_edge_id:
        tmp = np.zeros([model.lay_0.n_e, 1 + 2*args.max_num_node], dtype=np.float32)
        k = 0
        for i in range(args.max_num_node):
            for j in range(max(0, i-args.max_prev_node), i):
                tmp[k, 1 + j] = 1
                tmp[k, 1 + args.max_num_node + i] = 1
                k += 1
        assert k == model.lay_0.n_e
        input_edges = torch.tensor(np.tile(np.expand_dims(tmp, axis=0), [args.test_batch_size, 1, 1])).to(args.device)
    else:
        input_edges = torch.zeros(args.test_batch_size, first_layer.n_e, first_layer.num_input_features_edges).to(args.device)

    e = 0
    for i in range(args.max_num_node):
        # print('           i:', i)
        t = min(i, args.max_prev_node)
        for j in range(t):
            # print('                     j:', j)
            _, output_edges = model(input_nodes, input_edges)
            ind = torch.lt( torch.rand(args.test_batch_size).to(args.device), output_edges[:,e,0])
            input_edges[ind,e,0] = 1
            e += 1
        output_nodes, _ = model(input_nodes, input_edges)
        ind = torch.lt( torch.rand(args.test_batch_size).to(args.device), output_nodes[:,i,0])
        input_nodes[ind,i,0] = 1
        # print(i, ind.sum().item(), input_nodes[:,i,0].sum().item())
        # print(output_nodes[:,i,0], '\n')
        # input()

    # save graphs as pickle
    G_pred_list = []
    for i in range(args.test_batch_size):
        adj_pred = data.my_decode_adj(input_nodes[i,:,0].cpu().numpy(), input_edges[i,:,0].cpu().numpy(), args.max_prev_node)
        G_pred = utils.get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list



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
            # print(' #', i)
            print('.', end='')
            sys.stdout.flush()
            input_nodes = data['input_nodes_features'].float().to(args.device)
            input_edges = data['input_edges_features'].float().to(args.device)
            len_ = data['len'].float().to(args.device)

            optimizer.zero_grad()
            pred_nodes, pred_edges = gcade_model(input_nodes, input_edges)
            loss = nll(pred_nodes, pred_edges, input_nodes, input_edges, len_, args, i)
            # print('  ', loss.item() / input_nodes.size(0))
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            trsz += input_nodes.size(0)

        print('[epoch %d]     loss: %.3f                lr: %f' %
              (epoch + 1, running_loss / trsz, get_lr(optimizer)))
        time_end = time.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1,2): #4):
                print('     sample_time:', sample_time)
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    print('        len(G_pred):', len(G_pred))
                    G_pred_step = generate_graph(gcade_model, args)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                utils.save_graph_list(G_pred, fname)
            print('test done, graphs saved')

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
