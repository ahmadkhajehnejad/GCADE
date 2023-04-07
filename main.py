import create_graphs
import os
import random
from data import MyGraph_sequence_sampler_pytorch
from config import Args
# from model import GCADEModel, train
import torch
from transformer.Models import BlockWiseTransformer
import torch.optim as optim
from transformer.Optim import MyScheduledOptim #, ScheduledOptim
from utils import save_graph_list
import pickle
import argparse
from main_functions import train, just_test, just_generate

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# if __name__ == '__main__':

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random.seed(123)
# np.random.seed(123)
# torch.manual_seed(123)

args = Args()

graphs = create_graphs.create(args)   ## do not comment this line when use_pre_savede_graphs is True. This line sets args.max_prev_node too.

if args.use_pre_saved_graphs:

    with open(args.graph_save_path + args.fname_test + '0.dat', 'rb') as fin:
        graphs = pickle.load(fin)

    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

else:
    random.shuffle(graphs)

graphs_len = len(graphs)
graphs_test = graphs[int((1 - args.test_portion) * graphs_len):]
graphs_train = graphs[0:int(args.training_portion * graphs_len)]
graphs_validate = graphs[int((1 - args.test_portion - args.validation_portion) * graphs_len):
                         int((1 - args.test_portion) * graphs_len)]

if not args.use_pre_saved_graphs:
    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

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

max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
min_num_node = min([graphs[i].number_of_nodes() for i in range(len(graphs))])
max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

# show graphs statistics
print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
print('max number node: {}'.format(max_num_node))
print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
print('max previous node: {}'.format(args.max_prev_node))

args.num_blocks = int((max_num_node + 1 + (args.block_size - 1) - 1) / (args.block_size - 1))  # one element for termination bit, one element for shift right
args.max_num_node = args.num_blocks * (args.block_size - 1) - 1 # MyGraph_sequence_sampler_pytorch.__get_item__() will add one element to the end of the sequence
args.max_seq_len = args.max_num_node + 2
args.min_num_node = min_num_node

print('block_size:', args.block_size)
print('num_blocks:', args.num_blocks)

print('args.max_num_node:', args.max_num_node)
print('args.seq_len:', args.max_seq_len)

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

dataset = MyGraph_sequence_sampler_pytorch(graphs_train, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                 num_samples=args.batch_size * args.batch_ratio,
                                                                 replacement=True)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, #num_workers=args.num_workers,
                                             sampler=sample_strategy)

val_dataset = MyGraph_sequence_sampler_pytorch(graphs_validate, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
# val_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(val_dataset) for i in range(len(val_dataset))],
#                                                                  num_samples=args.batch_size * args.batch_ratio,
#                                                                  replacement=True)
val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size // 2, #num_workers=args.num_workers,
                                             sampler=None) #val_sample_strategy)


test_dataset = MyGraph_sequence_sampler_pytorch(graphs_test, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
# test_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(test_dataset) for i in range(len(test_dataset))],
#                                                                  num_samples=args.batch_size * args.batch_ratio,
#                                                                  replacement=True)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size // 2,  #num_workers=args.num_workers,
                                             sampler=None) #test_sample_strategy)


args.vocab_size = None

print('Preparing dataset finished.')

# gcade_model = GCADEModel(args)

# print('Model initiated.')

# train(gcade_model, dataset_loader, args)

model = BlockWiseTransformer(args).to(args.device)

print(model)

print('model initiated.')

# optimizer = ScheduledOptim(
#     optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#     args.lr_mul, args.d_model, args.n_warmup_steps)
optimizer = MyScheduledOptim(
    optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    None,
    args.milestones, args.lr_list, args.sep_optimizer_start_step)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)



'''
while True:
    for i, data in enumerate(test_dataset_loader):
        adj = data['adj'].to(args.device)
        print('           ##', i, '  ', adj[0, 0, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 1, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 2, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 3, :].sum().item(), '  ', adj.size(0))
        print('           ##', i, '  ', adj[0, 4, :].sum().item(), '  ', adj.size(0))
    print()
    input()
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', help='number of generation iteration (just generate)', action='store', type=int)
    parser.add_argument('-v', '--validate', help='just validate', action='store_true')
    parser.add_argument('-t', '--test', help='just test', action='store_true')
    console_args = parser.parse_args()

    if console_args.generate is not None:
        just_generate(model, dataset_loader, args, console_args.generate)
    elif console_args.validate:
        just_test(model, val_dataset_loader, args)
    elif console_args.test:
        just_test(model, test_dataset_loader, args)
    else:
        train(model, dataset_loader, val_dataset_loader, test_dataset_loader, optimizer, args)
