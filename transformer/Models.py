''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from made.made import MADE
from utils import prepare_for_MADE
from transformer.utils import compute_graph_kernels



__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    tmp = ((seq == pad_idx).sum(-1) == 0).unsqueeze(-2)
    return tmp.repeat(1, tmp.size(-1), 1)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    # subsequent_mask = torch.triu(subsequent_mask, diagonal=-40)
    return subsequent_mask


def outputPositionalEncoding(data, encType):
    seq_len = data.size(1)
    if encType == 'one_hot':
        return torch.eye(seq_len, device=data.device).unsqueeze(0).repeat(data.size(0), 1, 1)
    elif encType == 'tril':
        return torch.tril( torch.ones(data.size(0), seq_len, seq_len, device=data.device), diagonal=0)
    else:
        raise Exception('Unknown outputPositionalEncoding type:' + str(encType))

def get_lengths(seq, args, binary_nums):
    if len(seq.size()) == 3:
        seq = seq.unsqueeze(-2)
    tmp = seq[:,:,0,0] == args.one_input
    cnt = torch.arange(0,tmp.size(1)).long().to(args.device).reshape(1,-1).repeat(tmp.size(0), 1)
    ind  = cnt[tmp].reshape(-1)
    res = binary_nums[ind, :]
    return res.reshape(seq.size(0), -1)

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

class BasePositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(BasePositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class PositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(PositionalEncoding, self).__init__()
        self.base_positional_encoding = BasePositionalEncoding(args, d_hid, n_position)

    def forward(self, x):
        base_emb = self.base_positional_encoding(x)
        if len(x.size()) == 4:
            base_emb = base_emb.unsqueeze(-2)
        return x + base_emb


class GraphPositionalEncoding(nn.Module):

    def __init__(self, args, d_hid):
        super(GraphPositionalEncoding, self).__init__()
        if args.normalize_graph_positional_encoding:
            k_gr_kernel = 2 * args.k_graph_positional_encoding + 1
        else:
            k_gr_kernel = args.k_graph_positional_encoding + 1
        self.batchnorm = args.batchnormalize_graph_positional_encoding
        if self.batchnorm:
            self.layer_norm = nn.LayerNorm(k_gr_kernel, eps=1e-3)
        self.linear_1 = nn.Linear(k_gr_kernel, k_gr_kernel, bias=True)
        self.linear_2 = nn.Linear(k_gr_kernel, 1, bias=True)
        self.prj = nn.Linear(args.block_size, d_hid, bias = True)

    def forward(self, x, gr_kernel):
        input = gr_kernel.transpose(-3, -2).transpose(-2,-1)
        if self.batchnorm:
            input = self.layer_norm(input)
        gr_pos_enc = torch.sigmoid(self.linear_2(F.relu(self.linear_1(input))))
        gr_pos_enc_prj = self.prj(gr_pos_enc.squeeze(-1))
        if len(x.size()) == 4:
            gr_pos_enc_prj = gr_pos_enc_prj.unsqueeze(2)
        return x + gr_pos_enc_prj


class PropagationGraphPositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(PropagationGraphPositionalEncoding, self).__init__()
        self.base_positional_encoding = BasePositionalEncoding(args, d_hid, n_position)
        self.is_normalized = args.normalize_new_graph_positional_encoding
        self.eps = args.graph_positional_embedding_eps

    def forward(self, x, gr_kernel):
        batch_size = x.size(0)
        base_emb = self.base_positional_encoding(x).repeat(batch_size, 1, 1)
        coef = 1
        sum_coef = 0
        n_k = gr_kernel.size(1) - 1 if not self.is_normalized else int((gr_kernel.size(1) + 1) / 2) - 1
        for k in range(n_k + 1):
            sum_coef = sum_coef + coef
            tmp = torch.matmul( gr_kernel[:, k, :, :], base_emb) * coef
            if k == 0:
                emb = tmp
            else:
                emb = emb + tmp

            if self.is_normalized and k > 0:
                sum_coef = sum_coef + coef
                tmp = torch.matmul(gr_kernel[:, n_k + k, :, :], base_emb) * coef
                emb = emb + tmp

            coef = coef * (1 - self.eps)

        emb = emb / sum_coef

        if len(x.size()) == 4:
            emb = emb.unsqueeze(2)
        return x + emb


# class Encoder(nn.Module):
#     ''' A encoder model with self attention mechanism. '''
#
#     def __init__(self, args):
#
#         super().__init__()
#
#         self.args = args
#         sz_input_vec = args.n_ensemble * args.block_size
#         sz_emb = args.n_ensemble * args.d_word_vec
#
#         sz_intermed = max(sz_input_vec, sz_emb)
#         self.enc_word_emb_1 = nn.Linear(sz_input_vec, sz_intermed, bias=True)
#         self.enc_word_emb_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
#         self.enc_word_emb_3 = nn.Linear(sz_intermed, sz_emb, bias=True)
#
#         if args.k_graph_positional_encoding > 0:
#             if args.type_graph_positional_encoding == 1:
#                 self.position_enc = GraphPositionalEncoding(args=args, d_hid=args.d_word_vec)
#             elif args.type_graph_positional_encoding == 2:
#                 self.position_enc = PropagationGraphPositionalEncoding(args=args, d_hid=args.d_word_vec)
#             else:
#                 raise NotImplementedError()
#         else:
#             self.position_enc = PositionalEncoding(args=args, d_hid=args.d_word_vec, n_position=args.n_position)
#         self.dropout = nn.Dropout(p=args.dropout)
#         k_graph_attention = 2 * args.k_graph_attention + 1 if args.normalize_graph_attention else args.k_graph_attention + 1
#
#         self.layer_stack = nn.ModuleList([
#             EncoderLayer(args.d_model, args.d_inner_hid, args.n_ensemble, args.n_head, args.d_k, args.d_v, no_layer_norm=args.no_model_layer_norm,
#                          typed_edges=args.typed_edges, k_gr_att=k_graph_attention, gr_att_v2=args.graph_attention_version_2,
#                          gr_att_batchnorm=args.batchnormalize_graph_attention, dropout=args.dropout)
#             for _ in range(args.n_layers)])
#
#         if not args.no_model_layer_norm:
#             self.layer_norm = nn.LayerNorm(args.d_model, eps=1e-6)
#
#
#     def forward(self, enc_in, enc_in_mask, gr_mask, adj, gr_pos_enc_kernel, return_attns=False):
#
#         enc_slf_attn_list = []
#
#         # -- Forward
#         if len(enc_in.size()) == 4:
#             src_seq_tmp = enc_in.view(enc_in.size(0), enc_in.size(1), -1)
#         else:
#             src_seq_tmp = enc_in
#
#         enc_output = self.enc_word_emb_1(src_seq_tmp)
#         enc_output = self.enc_word_emb_2(nn.functional.relu(enc_output))
#         enc_output = self.enc_word_emb_3(nn.functional.relu(enc_output))
#         if len(enc_in.size()) == 4:
#             enc_output = enc_output.view(enc_in.size(0), enc_in.size(1), enc_in.size(2), -1)
#
#         if self.args.scale_emb:
#             enc_output *= self.args.d_model ** 0.5
#
#         if self.args.k_graph_positional_encoding > 0:
#             enc_output = self.dropout(self.position_enc(enc_output, gr_pos_enc_kernel))
#         else:
#             enc_output = self.dropout(self.position_enc(enc_output))
#         if not self.args.no_model_layer_norm:
#             enc_output = self.layer_norm(enc_output)
#
#         for i, enc_layer in enumerate(self.layer_stack):
#             enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_in_mask, gr_mask=gr_mask, adj=adj)
#             enc_slf_attn_list += [enc_slf_attn] if return_attns else []
#
#         if return_attns:
#             return enc_output, enc_slf_attn_list
#
#         return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, args):

        super().__init__()

        self.args = args
        sz_input_vec = args.n_ensemble * (args.max_num_node + 1)
        sz_emb = args.n_ensemble * args.d_word_vec

        sz_intermed = max(sz_input_vec, sz_emb)
        # sz_intermed = min(sz_input_vec, 2 * sz_emb)
        self.dec_word_emb_1 = nn.Linear(sz_input_vec, sz_intermed, bias=True)
        self.dec_word_emb_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
        self.dec_word_emb_3 = nn.Linear(sz_intermed, sz_emb, bias=True)

        if args.k_graph_positional_encoding > 0:
            if args.type_graph_positional_encoding == 1:
                self.position_enc = GraphPositionalEncoding(args=args, d_hid=args.d_word_vec)
            elif args.type_graph_positional_encoding == 2:
                self.position_enc = PropagationGraphPositionalEncoding(args=args, d_hid=args.d_word_vec)
            else:
                raise NotImplementedError()
        else:
            self.position_enc = PositionalEncoding(args=args, d_hid=args.d_word_vec, n_position=args.n_position)

        self.dropout = nn.Dropout(p=args.dropout)
        k_graph_attention = 2 * args.k_graph_attention + 1 if args.normalize_graph_attention else args.k_graph_attention + 1

        self.layer_stack = nn.ModuleList([
            DecoderLayer(args.d_model, args.d_inner_hid, args.n_ensemble, args.n_head, args.d_k, args.d_v, no_layer_norm=args.no_model_layer_norm,
                         typed_edges=args.typed_edges, k_gr_att=k_graph_attention, gr_att_v2=args.graph_attention_version_2,
                         gr_att_batchnorm=args.batchnormalize_graph_attention, dropout=args.dropout)
            for _ in range(args.n_layers)])

        if not args.no_model_layer_norm:
            self.layer_norm = nn.LayerNorm(args.d_model, eps=1e-6)


    def forward(self, dec_in, dec_in_mask, prev_dec_out, prev_dec_out_mask, gr_mask, adj, gr_pos_enc_kernel, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        if len(dec_in.size()) == 4:
            trg_seq_tmp = dec_in.view(dec_in.size(0), dec_in.size(1), -1)
        else:
            trg_seq_tmp = dec_in

        dec_output = self.dec_word_emb_1(trg_seq_tmp)
        dec_output = self.dec_word_emb_2(nn.functional.relu(dec_output))
        dec_output = self.dec_word_emb_3(nn.functional.relu(dec_output))
        if len(dec_in.size()) == 4:
            dec_output = dec_output.view(dec_in.size(0), dec_in.size(1), dec_in.size(2), -1)

        if self.args.scale_emb:
            dec_output *= self.args.d_model ** 0.5

        if self.args.k_graph_positional_encoding > 0:
            dec_output = self.dropout(self.position_enc(dec_output, gr_pos_enc_kernel))
        else:
            dec_output = self.dropout(self.position_enc(dec_output))

        if not self.args.no_model_layer_norm:
            dec_output = self.layer_norm(dec_output)

        prfx = ' '
        for dec_layer in self.layer_stack:
            prfx = prfx + ' '
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, prev_dec_out, slf_attn_mask=dec_in_mask, dec_enc_attn_mask=prev_dec_out_mask, gr_mask=gr_mask, adj=adj)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, args):

        super().__init__()

        if args.estimate_num_nodes:
            self.num_nodes_prob = None

        if args.weight_positions:
            self.positions_weights = None

        self.args = args

        # self.encoder = Encoder(args)

        self.decoder = Decoder(args)

        sz_out = args.max_num_node + 1

        sz_in = args.d_model * args.n_ensemble

        if args.feed_graph_length:
            sz_in = sz_in + int(np.ceil(np.log2(args.max_num_node + 1)))

        sz_intermed = max(sz_in, sz_out)

        if args.use_MADE:
            hidden_sizes = [sz_intermed * 3 // 2] * args.MADE_num_hidden_layers
            # hidden_sizes = [sz_out * 3 // 2] * args.MADE_num_hidden_layers

            self.trg_word_MADE = MADE(sz_in, hidden_sizes, sz_out, num_masks=args.MADE_num_masks,
                                      natural_ordering=args.MADE_natural_ordering)
        else:
            # self.trg_word_prj = nn.Linear(sz_in, sz_out, bias=False)
            self.trg_word_prj_1 = nn.Linear(sz_in, sz_intermed, bias=True)
            self.trg_word_prj_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
            self.trg_word_prj_3 = nn.Linear(sz_intermed, sz_out, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert args.d_model == args.d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'



    def forward(self, prev_dec_out, dec_in, trg, adj, graph_length, gr_mask, gr_pos_enc_kernel):

        if len(prev_dec_out.size()) == 3:
            prev_dec_out = prev_dec_out.unsqueeze(2).repeat(1, 1, self.args.n_ensemble, 1)
        if len(dec_in.size()) == 3:
            dec_in = dec_in.unsqueeze(2).repeat(1, 1, self.args.n_ensemble, 1)

        prev_dec_out_mask = get_pad_mask(prev_dec_out[:, :, 0, :], self.args.pad_idx)
        dec_in_mask = get_pad_mask(dec_in[:, :, 0, :], self.args.pad_idx) & get_subsequent_mask(dec_in[:, :, 0, :])

        dec_output, *_ = self.decoder(dec_in, dec_in_mask, prev_dec_out, prev_dec_out_mask, gr_mask, adj, gr_pos_enc_kernel)
        dec_output = dec_output.reshape(dec_output.size(0), dec_output.size(1), self.args.n_ensemble * self.args.d_model)

        made_axiliary_in = torch.cat([graph_length, dec_output], dim=2) if self.args.feed_graph_length else dec_output

        if self.args.use_MADE:
            seq_logit = self.trg_word_MADE(torch.cat([made_axiliary_in, prepare_for_MADE(trg, self.args)], dim=2))
        else:
            # seq_logit = self.trg_word_prj(made_axiliary_in)
            seq_logit = self.trg_word_prj_1(made_axiliary_in)
            seq_logit = self.trg_word_prj_2(nn.functional.relu(seq_logit))
            seq_logit = self.trg_word_prj_3(nn.functional.relu(seq_logit))

        if self.args.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit, dec_output, made_axiliary_in
        # return seq_logit.view(-1, seq_logit.size(2)), dec_output


class BlockWiseTransformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.shared_blocks:
            tmp = Transformer(args).to(args.device)
            blocks_list = [tmp for _ in range(args.num_blocks)]
        else:
            blocks_list = [Transformer(args).to(args.device) for _ in range(args.num_blocks)]
        self.blocks_stack = nn.ModuleList(blocks_list).to(args.device)

        if args.feed_graph_length:
            l = int(np.ceil(np.log2(args.max_num_node + 1)))
            self.binary_nums = binary( torch.arange(0, args.max_num_node + 1).int().to(args.device), l)


    def forward(self, dec_in, trg, adj):

        gr_mask, gr_pos_enc_kernel = compute_graph_kernels(adj, self.args)

        tmp_sz = list(dec_in.size())
        tmp_sz[1] = self.args.block_size
        tmp_sz[-1] = self.args.d_model
        tmp_prev_dec_out = torch.ones(tmp_sz, device=self.args.device)

        if self.args.feed_graph_length:
            tmp = dec_in[:, :, 0] == self.args.one_input
            graph_length = get_lengths(dec_in, self.args, self.binary_nums).float()
            graph_length = graph_length.unsqueeze(-2).repeat(1, self.args.block_size, 1)
        else:
            graph_length = None


        seq_logit_list = []
        dec_out_list =[]
        made_axiliary_in_list = []
        for block_i in range(self.args.num_blocks):

            first_idx = block_i * (self.args.block_size - 1)
            last_idx = (block_i + 1) * (self.args.block_size - 1) + 1

            tmp_dec_in = dec_in[:, first_idx: last_idx].clone()
            tmp_dec_in[:, 0] = self.args.pad_idx

            tmp_trg = trg[:, first_idx: last_idx].clone()
            tmp_trg[:, -1] = self.args.pad_idx

            tmp_adj = adj[:, first_idx : last_idx, first_idx : last_idx].clone()
            tmp_adj[0, :] = 0
            tmp_adj[:, 0] = 0

            if gr_mask is None:
                tmp_gr_mask = None
            else:
                tmp_gr_mask = gr_mask[ :, :, first_idx: last_idx, first_idx: last_idx]
                tmp_gr_mask[:, :, 0, :] = gr_mask[:, :, 0, first_idx: last_idx]
                tmp_gr_mask[:, :, :, 0] = gr_mask[:, :, first_idx: last_idx, 0]

            if gr_pos_enc_kernel is None:
                tmp_gr_pos_enc_kernel = None
            else:
                tmp_gr_pos_enc_kernel = gr_pos_enc_kernel[:, :, first_idx: last_idx, first_idx: last_idx]
                tmp_gr_pos_enc_kernel[:, :, 0, :] = gr_pos_enc_kernel[:, :, 0, first_idx: last_idx]
                tmp_gr_pos_enc_kernel[:, :, :, 0] = gr_pos_enc_kernel[:, :, first_idx: last_idx, 0]

            tmp_seq_logit, tmp_dec_out, tmp_made_axiliary_in = self.blocks_stack[block_i](tmp_prev_dec_out, tmp_dec_in,
                                                                                          tmp_trg, tmp_adj, graph_length,
                                                                                          tmp_gr_mask, tmp_gr_pos_enc_kernel)
            if block_i < self.args.num_blocks - 1:
                seq_logit_list.append(tmp_seq_logit[:, :-1])
                dec_out_list.append(tmp_dec_out[:, :-1])
                made_axiliary_in_list.append(tmp_made_axiliary_in[:, :-1])
            else:
                seq_logit_list.append(tmp_seq_logit)
                dec_out_list.append(tmp_dec_out)
                made_axiliary_in_list.append(tmp_made_axiliary_in)

            tmp_prev_dec_out = tmp_dec_out

        seq_logit = torch.cat(seq_logit_list, dim=1)
        dec_out = torch.cat(dec_out_list, dim=1)
        made_axiliary_in = torch.cat(made_axiliary_in_list, dim=1)

        return seq_logit, dec_out, made_axiliary_in

    def update_MADE_masks(self):
        if self.args.shared_blocks:
            self.blocks_stack[0].trg_word_MADE.update_masks()
        else:
            for t in self.blocks_stack:
                t.trg_word_MADE.update_masks()

    def get_MADE_output(self, made_in):
        made_out_list = []
        for block_i in range(self.args.num_blocks):
            first_idx = block_i * (self.args.block_size - 1)
            last_idx = (block_i + 1) * (self.args.block_size - 1)
            if block_i == self.args.num_blocks - 1:
                last_idx += 1
            tmp_made_in = made_in[:, first_idx: last_idx, :]
            made_out_list.append(self.blocks_stack[block_i].trg_word_MADE(tmp_made_in))
        return torch.cat( made_out_list, dim=1)
