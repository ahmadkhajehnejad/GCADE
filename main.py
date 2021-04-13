import create_graphs
import os
import random
from data import MyGraph_sequence_sampler_pytorch, my_decode_adj
from config import Args
# from model import GCADEModel, train
import numpy as np
import torch
from transformer.Models import Transformer
import torch.optim as optim
from transformer.Optim import ScheduledOptim
import time
import sys
import utils
import torch.nn.functional as F
from utils import save_graph_list


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# if __name__ == '__main__':

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

args = Args()
graphs = create_graphs.create(args)

# frq = np.zeros((21,))
# for g in graphs:
#     frq[len(g.nodes)] += 1
# print('\n'.join([str(k) + ' : ' + str(v) for k,v in {i:frq[i] for i in range(21) if frq[i] > 0}.items()]))
# input()

# split datasets
random.shuffle(graphs)
graphs_len = len(graphs)
graphs_test = graphs[int(0.8 * graphs_len):]
graphs_train = graphs[0:int(0.8 * graphs_len)]
graphs_validate = graphs[0:int(0.2 * graphs_len)]

# print([(len(list(g.nodes)), len(list(g.edges))) for g in graphs_train])
# print('---------------------------\n\n')
# input()

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

# save ground truth graphs
## To get train and test set, after loading you need to manually slice
save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

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
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             sampler=sample_strategy)

val_dataset = MyGraph_sequence_sampler_pytorch(graphs_validate, args, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
val_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(val_dataset) for i in range(len(val_dataset))],
                                                                 num_samples=args.batch_size * args.batch_ratio,
                                                                 replacement=True)
val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             sampler=val_sample_strategy)



if args.input_type == 'node_based':
    args.max_prev_node = dataset.max_prev_node
    args.max_seq_len = dataset.max_seq_len
    args.vocab_size = args.max_num_node + 3  # 0 for padding, self.n+1 for add_node, self.n+2 for termination
elif args.input_type == 'preceding_neighbors_vector':
    args.max_prev_node = dataset.max_prev_node
    args.max_seq_len = dataset.max_seq_len
    args.vocab_size = None
else:
    raise NotImplementedError

print('Preparing dataset finished.')

# gcade_model = GCADEModel(args)

# print('Model initiated.')

# train(gcade_model, dataset_loader, args)

model = Transformer(
    args.vocab_size,
    args.vocab_size,
    src_pad_idx=args.src_pad_idx,
    trg_pad_idx=args.trg_pad_idx,
    args=args,
    trg_emb_prj_weight_sharing=args.proj_share_weight,
    emb_src_trg_weight_sharing=args.embs_share_weight,
    d_k=args.d_k,
    d_v=args.d_v,
    d_model=args.d_model,
    d_word_vec=args.d_word_vec,
    d_inner=args.d_inner_hid,
    n_layers=args.n_layers,
    n_ensemble=args.n_ensemble,
    n_head=args.n_head,
    dropout=args.dropout,
    scale_emb_or_prj=args.scale_emb_or_prj).to(args.device)

print('model initiated.')

optimizer = ScheduledOptim(
    optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    args.lr_mul, args.d_model, args.n_warmup_steps)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)



def cal_performance(pred, gold, trg_pad_idx, args, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, args, smoothing=smoothing)
    if args.input_type == 'node_based':
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word
    elif args.input_type == 'preceding_neighbors_vector':
        return loss, None
    else:
        raise NotImplementedError


def cal_loss(pred, gold, trg_pad_idx, args, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    if smoothing:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            raise NotImplementedError
    else:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        elif args.input_type == 'preceding_neighbors_vector':

            pred = torch.sigmoid(pred).view(-1, args.max_seq_len, pred.size(-1))
            # gold = gold.view(-1, args.max_seq_len, gold.size(-1))

            # print('\n', pred)

            # gold = gold.clone()
            # ind_0 = gold == args.zero_input
            # ind_1 = gold == args.one_input
            # ind_2 = gold == args.dontcare_input
            # ind_3 = gold == args.trg_pad_idx
            # gold[ind_0] = 0
            # gold[ind_1] = 1
            # gold[ind_2] = 0
            # gold[ind_3] = -1

            cond_1 = gold != args.trg_pad_idx
            pred_1 = torch.tril(pred * cond_1, diagonal=0)
            gold_1 = torch.tril(gold * cond_1, diagonal=0)
            ind_0 = gold_1 == args.zero_input
            ind_1 = gold_1 == args.one_input
            gold_1[ind_0] = 0
            gold_1[ind_1] = 1

            loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='sum')

            cond_0 = gold[:,:,0] != args.trg_pad_idx
            cond_0[:, 0] = False
            cond_2 = cond_0.unsqueeze(-1).repeat(1, 1, gold.size(-1))
            pred_2 = torch.tril(pred * cond_2, diagonal=0)
            gold_2 = torch.zeros(gold.size(0), gold.size(1), gold.size(2), device=gold.device)

            # print(pred_1[0])

            p_zero = torch.exp(-F.binary_cross_entropy(pred_2, gold_2, reduction='none').sum(-1))
            loss_2 = torch.log(1-p_zero[cond_0]).sum()
            loss = loss_1 + loss_2
            # print('###############################   ', p_zero[cond_0].max().item())
            # print('#########################                ', loss_1.item(), loss_2.item(), loss.item())
            # input()
            # loss = loss_1
        else:
            raise NotImplementedError
    return loss

def generate_graph(gg_model, args):

    # return None
    gg_model.eval()

    if args.input_type == 'node_based':
        src_seq = torch.zeros((args.test_batch_size, args.max_seq_len), dtype=torch.long).to(args.device)
        for i in range(args.max_seq_len - 1):
            #pred = gg_model(src_seq, src_seq).max(1)[1].view([args.test_batch_size, args.max_seq_len])
            pred_logprobs = gg_model(src_seq, src_seq) #.max(1)[1].view([args.test_batch_size, args.max_seq_len])
            pred_probs = pred_logprobs.exp() / pred_logprobs.exp().sum(axis=-1, keepdim=True).repeat(1,pred_logprobs.size(-1))
            pred = torch.tensor([np.random.choice(np.arange(probs.size(0)),size=1,p=probs.detach().cpu().numpy())[0]
                                 for probs in pred_probs]).view([args.test_batch_size, args.max_seq_len]).to(args.device)
            src_seq[:, i + 1] = pred[:, i]
    elif args.input_type == 'preceding_neighbors_vector':
        src_seq = args.src_pad_idx * torch.ones((args.test_batch_size, args.max_seq_len, args.max_num_node + 1),
                                  dtype=torch.float32).to(args.device)
        not_finished_idx = torch.ones([src_seq.size(0)]).bool().to(args.device)
        for i in range(args.max_seq_len - 1):
            pred_probs = torch.sigmoid(gg_model(src_seq, src_seq, )).view(-1, args.max_seq_len, args.max_num_node + 1)
            num_trials = 0
            remainder_idx = not_finished_idx.clone()
            src_seq[remainder_idx, i+1, i+1:] = args.dontcare_input
            while remainder_idx.sum().item() > 0:
                num_trials += 1
                tmp = (torch.rand([remainder_idx.sum().item(), i + 1], device=args.device) < pred_probs[remainder_idx,
                                                                                         i, :i + 1]).float()
                ind_0 = tmp == 0
                ind_1 = tmp == 1
                tmp[ind_0] = args.zero_input
                tmp[ind_1] = args.one_input
                src_seq[remainder_idx, i + 1, :i + 1] = tmp
                if i == 0:
                    break
                remainder_idx = remainder_idx & ((src_seq[:, i + 1, : i + 1] == args.one_input).sum(-1) == 0)
            new_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] == args.one_input)
            src_seq[new_finished_idx, i + 1, 1:] = args.src_pad_idx
            not_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] != args.one_input)
            if num_trials > 1:
                print('                          ', i, '      num of trials:', num_trials)
            if not_finished_idx.sum().item() == 0:
                break

        ind_0 = src_seq == args.zero_input
        ind_1 = src_seq == args.one_input
        src_seq[ind_0] = 0
        src_seq[ind_1] = 1
        # for i in range(args.max_seq_len - 1):
        #     pred_probs = torch.sigmoid(gg_model(src_seq, src_seq)).view(-1, args.max_seq_len, args.max_num_node + 1)
        #     src_seq[:, i+1, :] = (torch.rand(pred_probs[:, i, :].size(), device=args.device) < pred_probs[:, i, :]).float()
    else:
        raise NotImplementedError


    # save graphs as pickle
    G_pred_list = []
    for i in range(args.test_batch_size):
        adj_pred = my_decode_adj(src_seq[i,1:].cpu().numpy(), args)
        G_pred = utils.get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def train(gg_model, dataset_train, dataset_validation, optimizer, args):

    ## initialize optimizer
    ## optimizer = torch.optim.Adam(list(gcade_model.parameters()), lr=args.lr)
    ## scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        time_start = time.time()
        running_loss = 0.0
        trsz = 0
        gg_model.train()
        for i, data in enumerate(dataset_train, 0):
            # print(' #', i)
            print('.', end='')
            sys.stdout.flush()
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device)
            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            optimizer.zero_grad()
            pred = gg_model(src_seq, trg_seq, adj)
            loss, *_ = cal_performance( pred, gold, trg_pad_idx=0, args=args, smoothing=False)
            # print('  ', loss.item() / input_nodes.size(0))
            loss.backward()
            optimizer.step_and_update_lr()

            running_loss += loss.item()
            trsz += src_seq.size(0)

        val_running_loss = 0.0
        vlsz = 0
        gg_model.eval()
        for i, data in enumerate(dataset_validation, 0):
            src_seq = data['src_seq'].to(args.device)
            trg_seq = data['src_seq'].to(args.device)
            gold = data['trg_seq'].contiguous().to(args.device)
            adj = data['adj'].to(args.device)

            pred = gg_model(src_seq, trg_seq, adj)
            loss, *_ = cal_performance( pred, gold, trg_pad_idx=0, args=args, smoothing=False)

            val_running_loss += loss.item()
            vlsz += src_seq.size(0)


        print('[epoch %d]     loss: %.3f     val: %.3f              lr: %f' %
              (epoch + 1, running_loss / trsz, val_running_loss / vlsz, optimizer._optimizer.param_groups[0]['lr'])) #get_lr(optimizer)))
        time_end = time.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1,2): #4):
                print('     sample_time:', sample_time)
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    print('        len(G_pred):', len(G_pred))
                    G_pred_step = generate_graph(gg_model, args)
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

train(model, dataset_loader, val_dataset_loader, optimizer, args)
