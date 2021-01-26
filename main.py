import create_graphs
import torch
import os
import random
from data import MyGraph_sequence_sampler_pytorch
from config import Args
import numpy as np
from model import GCADEModel, train


# if __name__ == '__main__':

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args()
graphs = create_graphs.create(args)

# split datasets
random.seed(123)
random.shuffle(graphs)
graphs_len = len(graphs)
graphs_test = graphs[int(0.8 * graphs_len):]
graphs_train = graphs[0:int(0.8 * graphs_len)]
graphs_validate = graphs[0:int(0.2 * graphs_len)]

# if use pre-saved graphs
# dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
# fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
#     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
# graphs = load_graph_list(fname_test, is_real=True)
# graphs_test = graphs[int(0.8 * graphs_len):]
# graphs_train = graphs[0:int(0.8 * graphs_len)]
# graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

graph_validate_len = 0
for graph in graphs_validate:
    graph_validate_len += graph.number_of_nodes()
graph_validate_len /= len(graphs_validate)
print('graph_validate_len', graph_validate_len)

graph_test_len = 0
for graph in graphs_test:
    graph_test_len += graph.number_of_nodes()
graph_test_len /= len(graphs_test)
print('graph_test_len', graph_test_len)

args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

# show graphs statistics
print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
print('max number node: {}'.format(args.max_num_node))
print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
print('max previous node: {}'.format(args.max_prev_node))

# # save ground truth graphs
# ## To get train and test set, after loading you need to manually slice
# save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
# save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
# print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

### comment when normal training, for graph completion only
# p = 0.5
# for graph in graphs_train:
#     for node in list(graph.nodes()):
#         # print('node',node)
#         if np.random.rand()>p:
#             graph.remove_node(node)
# for edge in list(graph.edges()):
#     # print('edge',edge)
#     if np.random.rand()>p:
#         graph.remove_edge(edge[0],edge[1])

### dataset initialization
dataset = MyGraph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)



sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                 num_samples=args.batch_size * args.batch_ratio,
                                                                 replacement=True)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             sampler=sample_strategy)


print('Preparing dataset finished.')

gcade_model = GCADEModel(args)

print('Model initiated.')

train(gcade_model, dataset_loader, args)
