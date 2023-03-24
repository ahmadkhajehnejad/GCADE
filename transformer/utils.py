import torch
import numpy as np


def compute_graph_kernels(adj, args):
    k_gr_att = args.k_graph_attention
    k_gr_pos_enc = args.k_graph_positional_encoding

    k_gr = max(k_gr_att, k_gr_pos_enc)

    if k_gr > 0:
        gr_paths_count = torch.zeros(adj.size(0), k_gr + 1, adj.size(1), adj.size(2)).to(args.device)
        gr_paths_count[:, 0, :, :] = torch.eye(adj.size(1)).to(args.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        gr_paths_count[:, 1, :, :] = torch.triu(adj)
        for i in range(2, k_gr + 1):
            gr_paths_count[:, i, :, :] = torch.triu(torch.matmul(adj, gr_paths_count[:, i - 1, :, :]))
        gr_paths_count = torch.transpose(gr_paths_count, 2, 3)

        if args.normalize_graph_attention or args.normalize_graph_positional_encoding:
            sm = gr_paths_count.sum(-1, keepdim=True)
            gr_normalized_paths_count_1 = gr_paths_count / sm.masked_fill(sm_1 == 0, 1)

            gr_all_paths_count = torch.zeros(adj.size(0), k_gr + 1, adj.size(1), adj.size(2)).to(args.device)
            gr_all_paths_count[:, 0, :, :] = torch.triu(adj)
            for i in range(1, k_gr + 1):
                gr_all_paths_count[:, i, :, :] = torch.triu(torch.matmul(adj, gr_all_paths_count[:, i - 1, :, :]))
            gr_all_paths_count = torch.transpose(gr_all_paths_count, 2, 3)
            gr_noramlized_paths_count_2 = gr_paths_count / gr_all_paths_count.masked_fill(gr_all_paths_count == 0, 1)

        if k_gr_att <= 0:
            gr_mask = None
        elif args.normalize_graph_attention:
            gr_mask = torch.cat([gr_normalized_paths_count_1[:, :k_gr_att + 1],
                                 gr_noramlized_paths_count_2[:, 1:k_gr_att + 1]], dim=1)
        elif args.log_graph_attention:
            gr_mask = torch.log(gr_paths_count[:, :k_gr_att + 1, :, :] + 1)
        else:
            gr_mask = gr_paths_count[:, :k_gr_att + 1, :, :]

        if k_gr_pos_enc <= 0:
            gr_pos_enc_kernel = None
        elif args.normalize_graph_positional_encoding:
            gr_pos_enc_kernel = torch.cat([gr_normalized_paths_count_1[:, :k_gr_pos_enc + 1],
                                 gr_noramlized_paths_count_2[:, 1:k_gr_pos_enc + 1]], dim=1)
        elif args.log_graph_positional_encoding:
            gr_pos_enc_kernel = torch.log(gr_paths_count[:, :k_gr_pos_enc + 1, :, :] + 1)
        else:
            gr_pos_enc_kernel = gr_paths_count[:, :k_gr_pos_enc + 1, :, :]

    else:
        gr_mask = None
        gr_pos_enc_kernel = None

    return gr_mask, gr_pos_enc_kernel
