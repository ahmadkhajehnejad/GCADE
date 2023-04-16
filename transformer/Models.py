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


def get_pad_mask(seq, pad_idx, input_type):
    if input_type == 'node_based':
        return (seq != pad_idx).unsqueeze(-2)
    elif input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
        return ((seq == pad_idx).sum(-1) == 0).unsqueeze(-2)
    else:
        raise NotImplementedError


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
    assert args.use_termination_bit
    tmp = seq[:,:,0,0] == args.one_input
    cnt = torch.arange(0,tmp.size(1)).long().to(args.device).reshape(1,-1).repeat(tmp.size(0), 1)
    ind  = cnt[tmp].reshape(-1)
    res = binary_nums[ind, :]
    return res.reshape(seq.size(0), 1, -1).repeat(1,seq.size(1),1)

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

class BasePositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(BasePositionalEncoding, self).__init__()

        self.input_type = args.input_type
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

    def forward(self, x, gr_kernel=None):
        return self.pos_table[:, :x.size(1)].clone().detach()


class PositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(PositionalEncoding, self).__init__()
        self.base_positional_encoding = BasePositionalEncoding(args, d_hid, n_position)

    def forward(self, x, gr_kernel=None):
        base_emb = self.base_positional_encoding(x)
        if len(x.size()) == 4:
            base_emb = base_emb.unsqueeze(-2)
        return x + base_emb


class BaseGraphPositionalEncoding(nn.Module):

    def __init__(self, args, d_hid):
        super(BaseGraphPositionalEncoding, self).__init__()
        if args.normalize_graph_positional_encoding:
            k_gr_kernel = 2 * args.k_graph_positional_encoding + 1
        else:
            k_gr_kernel = args.k_graph_positional_encoding + 1
        self.batchnorm = args.batchnormalize_graph_positional_encoding
        if self.batchnorm:
            self.layer_norm = nn.LayerNorm(k_gr_kernel, eps=1e-3)
        self.linear_1 = nn.Linear(k_gr_kernel, k_gr_kernel, bias=True)
        self.linear_2 = nn.Linear(k_gr_kernel, 1, bias=True)
        self.prj = nn.Linear(args.max_num_node + 2, d_hid, bias = True)

    def forward(self, x, gr_kernel):
        input = gr_kernel.transpose(-3, -2).transpose(-2,-1)
        if self.batchnorm:
            input = self.layer_norm(input)
        gr_pos_enc = torch.sigmoid(self.linear_2(F.relu(self.linear_1(input))))
        gr_pos_enc_prj = self.prj(gr_pos_enc.squeeze(-1))
        return gr_pos_enc_prj


class GraphPositionalEncoding(nn.Module):

    def __init__(self, args, d_hid):
        super(GraphPositionalEncoding, self).__init__()
        self.base_graph_positional_encoding = BaseGraphPositionalEncoding(args, d_hid)

    def forward(self, x, gr_kernel):
        base_emb = self.base_graph_positional_encoding(x, gr_kernel)
        if len(x.size()) == 4:
            base_emb = base_emb.unsqueeze(-2)
        return x + base_emb


class BasePropagationGraphPositionalEncoding(nn.Module):

    def __init__(self, args, base_positional_encoding):
        super(BasePropagationGraphPositionalEncoding, self).__init__()
        self.base_positional_encoding = base_positional_encoding
        self.is_normalized = args.normalize_new_graph_positional_encoding
        self.eps = args.graph_positional_embedding_eps

    def forward(self, x, gr_kernel):
        batch_size = x.size(0)
        base_emb = self.base_positional_encoding(x, gr_kernel)
        if base_emb.size(0) == 1:
            base_emb = base_emb.repeat(batch_size, 1, 1)
        coef = 1
        sum_coef = 0
        n_k = gr_kernel.size(1) - 1 if not self.is_normalized else int((gr_kernel.size(1) + 1) / 2) - 1
        emb = None
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

        return emb


class PropagationGraphPositionalEncoding(nn.Module):

    def __init__(self, args, base_positional_encoding):
        super(PropagationGraphPositionalEncoding, self).__init__()
        self.base_propagation_graph_positional_encoding = BasePropagationGraphPositionalEncoding(args, base_positional_encoding)

    def forward(self, x, gr_kernel):
        base_emb = self.base_propagation_graph_positional_encoding(x, gr_kernel)
        if len(x.size()) == 4:
            base_emb = base_emb.unsqueeze(-2)
        return x + base_emb

class NoAddPropagationGraphPositionalEncoding(nn.Module):

    def __init__(self, args, base_positional_encoding):
        super(NoAddPropagationGraphPositionalEncoding, self).__init__()
        self.base_propagation_graph_positional_encoding = BasePropagationGraphPositionalEncoding(args, base_positional_encoding)

    def forward(self, x, gr_kernel):
        base_emb = self.base_propagation_graph_positional_encoding(x, gr_kernel)
        if len(x.size()) == 4:
            base_emb = base_emb.unsqueeze(-2)
        return base_emb


# class NewGraphPositionalEncoding(nn.Module):
#
#     def __init__(self, args, d_hid):
#         super(NewGraphPositionalEncoding, self).__init__()
#         if args.normalize_new_graph_positional_encoding:
#             k_gr_kernel = 2 * args.k_new_graph_positional_encoding + 1
#         else:
#             k_gr_kernel = args.k_new_graph_positional_encoding + 1
#         self.batchnorm = args.batchnormalize_new_graph_positional_encoding
#         if self.batchnorm:
#             self.layer_norm = nn.LayerNorm(k_gr_kernel, eps=1e-3)
#         self.linear_1 = nn.Linear(k_gr_kernel, k_gr_kernel, bias=True)
#         self.linear_2 = nn.Linear(k_gr_kernel, 1, bias=True)
#         self.device = args.device
#
#         self.initial_embeddings = torch.tensor(np.random.rand(args.max_seq_len, d_hid), dtype=torch.float32, device=args.device)
#
#     def forward(self, x, gr_kernel):
#
#         input = gr_kernel.transpose(-3, -2).transpose(-2, -1)
#         if self.batchnorm:
#             input = self.layer_norm(input)
#         sim = torch.exp(self.linear_2(F.relu(self.linear_1(input))))
#         sim = sim.squeeze(-1)
#
#         batch_size, k, max_n, _ = gr_kernel.size()
#
#         emb = self.initial_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
#         for i in range(1, max_n):
#             tmp = torch.matmul(sim[:, i:i + 1, :i + 1], emb[:, :i + 1, :])
#             denominator = sim[:, i:i + 1, :i + 1].sum(dim=-1, keepdim=True).repeat(1, 1, emb.size(2))
#             denominator[denominator == 0] = 1
#             emb[:, i:i + 1, :] = tmp / denominator
#
#         if len(x.size()) == 4:
#             emb = emb.unsqueeze(2)
#         return x + emb


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_ensemble, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, args, dropout=0.1, n_position=1000, scale_emb=False):

        super().__init__()

        self.args = args
        if args.input_type == 'node_based':
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            if args.input_type == 'preceding_neighbors_vector':
                if args.ensemble_input_type == 'multihop-single':
                    sz_input_vec = (1 + len(args.ensemble_multihop)) * (args.max_num_node + 1)
                    sz_emb = d_word_vec
                else:
                    sz_input_vec = n_ensemble * (args.max_num_node + 1)
                    sz_emb = n_ensemble * d_word_vec
            else:
                if args.ensemble_input_type == 'multihop-single':
                    sz_input_vec = args.max_prev_node + 1 + len(args.ensemble_multihop) * (args.max_num_node)
                    sz_emb = d_word_vec
                else:
                    sz_input_vec = n_ensemble * (args.max_prev_node + 1)
                    sz_emb = n_ensemble * d_word_vec
            if args.input_bfs_depth:
                sz_input_vec = sz_input_vec + args.max_num_node
            # self.src_word_emb = nn.Linear(sz_input_vec, sz_emb, bias=False)
            sz_intermed = max(sz_input_vec, sz_emb)
            self.src_word_emb_1 = nn.Linear(sz_input_vec, sz_intermed, bias=True)
            self.src_word_emb_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
            self.src_word_emb_3 = nn.Linear(sz_intermed, sz_emb, bias=True)

        else:
            raise NotImplementedError
        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.n_layers = n_layers
        self.n_grlayers = args.n_grlayers
        if args.k_graph_positional_encoding > 0:
            if args.type_graph_positional_encoding == 1:
                self.position_enc = GraphPositionalEncoding(args=args, d_hid=args.d_word_vec)
            elif args.type_graph_positional_encoding == 2:
                self.position_enc = PropagationGraphPositionalEncoding(args=args,
                                                                       base_positional_encoding=BasePositionalEncoding(args,
                                                                                                                       d_hid=args.d_word_vec,
                                                                                                                       n_position=n_position))
            elif args.type_graph_positional_encoding == 3:
                self.position_enc = PropagationGraphPositionalEncoding(args=args,
                                                                       base_positional_encoding=BaseGraphPositionalEncoding(args,
                                                                                                                            d_hid=args.d_word_vec))
            elif args.type_graph_positional_encoding == 4:
                self.position_enc = NoAddPropagationGraphPositionalEncoding(args=args,
                                                                            base_positional_encoding=BaseGraphPositionalEncoding(args,
                                                                                                                                 d_hid=args.d_word_vec))
            else:
                raise NotImplementedError()
        else:
            self.position_enc = PositionalEncoding(args=args, d_hid=args.d_word_vec, n_position=args.n_position)

        self.dropout = nn.Dropout(p=dropout)
        k_graph_attention = 2 * args.k_graph_attention + 1 if args.normalize_graph_attention else args.k_graph_attention + 1

        self.num_shared_parameters = len(list(self.parameters()))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_ensemble, n_head, d_k, d_v, no_layer_norm=args.no_model_layer_norm,
                         typed_edges=args.typed_edges, k_gr_att=k_graph_attention, gr_att_v2=args.graph_attention_version_2,
                         gr_att_batchnorm=args.batchnormalize_graph_attention, dropout=dropout)
            for _ in range(n_layers)])

        if args.separate_termination_bit:
            self.num_shared_parameters += int( (len(list(self.layer_stack)) / n_layers) * (n_layers - args.sepTermBitNumLayers) )
        else:
            self.num_shared_parameters = len(list(self.layer_stack))

        if args.separate_termination_bit:
            assert args.only_encoder
            assert args.sepTermBitNumLayers < n_layers
            self.termination_bit_layer_stack = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_ensemble, n_head, d_k, d_v,
                             no_layer_norm=args.no_model_layer_norm,
                             typed_edges=args.typed_edges, k_gr_att=k_graph_attention,
                             gr_att_batchnorm=args.batchnormalize_graph_attention,
                             dropout=dropout)
                for _ in range(args.sepTermBitNumLayers)
            ])
            self.sepTermBitNumLayers = args.sepTermBitNumLayers
        self.separate_termination_bit = args.separate_termination_bit

        if not args.no_model_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model


    def forward(self, src_seq, src_mask, gr_mask, adj, gr_pos_enc_kernel, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        if len(src_seq.size()) == 4:
            src_seq_tmp = src_seq.view(src_seq.size(0), src_seq.size(1), -1)
        else:
            src_seq_tmp = src_seq
        # enc_output = self.src_word_emb(src_tmp)
        enc_output = self.src_word_emb_1(src_seq_tmp)
        enc_output = self.src_word_emb_2(nn.functional.relu(enc_output))
        enc_output = self.src_word_emb_3(nn.functional.relu(enc_output))
        if len(src_seq.size()) == 4:
            enc_output = enc_output.view(src_seq.size(0), src_seq.size(1), src_seq.size(2), -1)

        if self.scale_emb:
            enc_output *= self.d_model ** 0.5


        enc_output = self.dropout(self.position_enc(enc_output, gr_pos_enc_kernel))
        if not self.args.no_model_layer_norm:
            enc_output = self.layer_norm(enc_output)

        if self.n_grlayers > 0:
            gr_src_mask = torch.tril(adj, diagonal=0)

            diag_ind = torch.eye(gr_src_mask.size(1)).unsqueeze(0).repeat(gr_src_mask.size(0), 1, 1).bool().to(
                gr_src_mask.device)
            gr_src_mask[diag_ind] = 1

            gr_src_mask = gr_src_mask * src_mask   ## this line is not necessary and can be removed

        for i, enc_layer in enumerate(self.layer_stack):
            if i < self.n_grlayers:
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=gr_src_mask, gr_mask=None, adj=adj)
            else:
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask, gr_mask=gr_mask, adj=adj)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            if self.separate_termination_bit and i == len(self.layer_stack) - self.sepTermBitNumLayers - 1:
                enc_output_termination_bit = enc_output

        if self.separate_termination_bit:
            for i, sep_layer in enumerate(self.termination_bit_layer_stack):
                enc_output_termination_bit, enc_termbit_attn = sep_layer(enc_output_termination_bit, slf_attn_mask=src_mask, gr_mask=gr_mask, adj=adj)
                enc_slf_attn_list += [enc_termbit_attn] if return_attns else []

        if return_attns:
            if self.separate_termination_bit:
                return enc_output, enc_output_termination_bit, enc_slf_attn_list
            return enc_output, enc_slf_attn_list

        if self.separate_termination_bit:
            return enc_output, enc_output_termination_bit
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_ensemble, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, args, n_position=1000, dropout=0.1, scale_emb=False):

        super().__init__()

        if args.input_type == 'node_based':
            self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            if args.input_type == 'preceding_neighbors_vector':
                if args.ensemble_input_type == 'multihop-single':
                    sz_input_vec = (1 + len(args.ensemble_multihop)) * (args.max_num_node + 1)
                    sz_emb = d_word_vec
                else:
                    sz_input_vec = n_ensemble * (args.max_num_node + 1)
                    sz_emb = n_ensemble * d_word_vec
            else:
                if args.ensemble_input_type == 'multihop-single':
                    sz_input_vec = args.max_prev_node + 1 + len(args.ensemble_multihop) * (args.max_num_node)
                    sz_emb = d_word_vec
                else:
                    sz_input_vec = n_ensemble * (args.max_prev_node + 1)
                    sz_emb = n_ensemble * d_word_vec
            # self.trg_word_emb = nn.Linear(sz_input_vec, sz_emb, bias=False)
            sz_intermed = max(sz_input_vec, sz_emb)
            self.trg_word_emb_1 = nn.Linear(sz_input_vec, sz_intermed, bias=True)
            self.trg_word_emb_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
            self.trg_word_emb_3 = nn.Linear(sz_intermed, sz_emb, bias=True)
        else:
            raise NotImplementedError
        # self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(args=args, d_hid=d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        k_graph_attention = 2 * args.k_graph_attention + 1 if args.normalize_graph_attention else args.k_graph_attention + 1
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_ensemble, n_head, d_k, d_v, no_layer_norm=args.no_model_layer_norm,
                         typed_edges=args.typed_edges, k_gr_att=k_graph_attention, gr_att_v2=args.graph_attention_version_2,
                         gr_att_batchnorm=args.batchnormalize_graph_attention, dropout=dropout)
            for _ in range(n_layers)])
        if not args.no_model_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, gr_mask, adj, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        if len(trg_seq.size()) == 4:
            trg_seq_tmp = trg_seq.view(trg_seq.size(0), trg_seq.size(1), -1)
        else:
            trg_seq_tmp = trg_seq
        # dec_output = self.trg_word_emb(trg_tmp)
        dec_output = self.trg_word_emb_1(trg_seq_tmp)
        dec_output = self.trg_word_emb_2(nn.functional.relu(dec_output))
        dec_output = self.trg_word_emb_3(nn.functional.relu(dec_output))
        if len(trg_seq.size()) == 4:
            dec_output = dec_output.view(trg_seq.size(0), trg_seq.size(1), trg_seq.size(2), -1)

        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        if not self.args.no_model_layer_norm:
            dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask, gr_mask=gr_mask, adj=adj)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx, args,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_ensemble=8, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=1000,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        if args.feed_graph_length:
            l = int(np.ceil(np.log2(args.max_seq_len)))
            self.binary_nums = binary( torch.arange(0, args.max_seq_len).int().to(args.device), l)

        if args.estimate_num_nodes:
            self.num_nodes_prob = None
        if args.weight_positions:
            self.positions_weights = None
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.args = args

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        self.n_ensemble = n_ensemble
        self.separate_termination_bit = args.separate_termination_bit

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_ensemble=n_ensemble, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, args=args, dropout=dropout, scale_emb=scale_emb)

        if not args.only_encoder:
            self.decoder = Decoder(
                n_trg_vocab=n_trg_vocab, n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_ensemble=n_ensemble, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=trg_pad_idx, args=args, dropout=dropout, scale_emb=scale_emb)

        if args.input_type == 'node_based':
            self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:

            if args.input_type == 'preceding_neighbors_vector':
                sz_out = args.max_num_node + 1
            else:
                sz_out = args.max_prev_node + 1
            if args.separate_termination_bit:
                sz_out = sz_out - 1

            sz_in = d_model * n_ensemble

            if args.use_MADE:
                sz_in_new = max(self.d_model // args.MADE_dim_reduction_factor, 10)
                # self.before_trg_word_MADE = nn.Linear(sz_in, sz_in_new)
                # self.before_trg_word_MADE = nn.Sequential(*[nn.Linear(sz_in, sz_in), nn.ReLU(), nn.Linear(sz_in, sz_in), nn.ReLU(), nn.Linear(sz_in, sz_in_new)])
                sz_in = sz_in_new
                # self.before_MADE_norm = nn.LayerNorm(sz_in_new, eps=1e-6)

            if args.output_positional_embedding:
                sz_in = sz_in + args.max_seq_len

            if args.feed_graph_length:
                sz_in = sz_in + int(np.ceil(np.log2(args.max_seq_len)))

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

            if args.separate_termination_bit:
                sz_in_term = d_model * n_ensemble
                self.termination_bit_prj_1 = nn.Linear(sz_in_term, sz_in_term, bias=True)
                self.termination_bit_prj_2 = nn.Linear(sz_in_term, sz_in_term, bias=True)
                self.termination_bit_prj_3 = nn.Linear(sz_in_term, 1, bias=True)

        else:
            raise NotImplementedError

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            if args.input_type == 'node_based':
                self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            else:
                raise NotImplementedError  #TODO: implement it for 'preceding_neighbors_vector' input type

        if (not args.only_encoder) and emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


        # for p in self.encoder.parameters():
        #     print(p.size())
        # input()

    def forward(self, src_seq, trg_seq, gold, adj):

        gr_mask, gr_pos_enc_kernel = compute_graph_kernels(adj, self.args)

        if len(src_seq.size()) == 4:
            src_mask = get_pad_mask(src_seq[:, :, 0, :], self.args.src_pad_idx,
                                    self.args.input_type) & get_subsequent_mask(src_seq[:, :, 0, :])
        else:
            src_mask = get_pad_mask(src_seq, self.args.src_pad_idx, self.args.input_type) & get_subsequent_mask(src_seq)

        if len(trg_seq.size()) == 4:
            trg_mask = get_pad_mask(trg_seq[:, :, 0, :], self.args.src_pad_idx,
                                    self.args.input_type) & get_subsequent_mask(trg_seq[:, :, 0, :])
        else:
            trg_mask = get_pad_mask(trg_seq, self.args.src_pad_idx, self.args.input_type) & get_subsequent_mask(trg_seq)

        if self.args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            if len(src_seq.size()) == 3:
                src_seq = src_seq.unsqueeze(2).repeat(1, 1, self.n_ensemble, 1)
            if len(trg_seq.size()) == 3:
                trg_seq = trg_seq.unsqueeze(2).repeat(1, 1, self.n_ensemble, 1)

        if self.separate_termination_bit:
            enc_output, semifinal_enc_output, *_ = self.encoder(src_seq, src_mask, gr_mask, adj, gr_pos_enc_kernel)
        else:
            enc_output, *_ = self.encoder(src_seq, src_mask, gr_mask, adj, gr_pos_enc_kernel)

        if self.args.only_encoder:
            dec_output = enc_output
        else:
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask, gr_mask, adj)


        if self.args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            dec_output = dec_output.reshape(dec_output.size(0), dec_output.size(1), self.n_ensemble * self.d_model)
            # if self.args.use_MADE:
                # dec_output = torch.sigmoid(self.before_trg_word_MADE(dec_output))
                # dec_output = torch.relu(self.before_trg_word_MADE(dec_output))
                # dec_output = torch.sigmoid(self.before_trg_word_MADE(dec_output)) * 2 - 1
                # dec_output = self.before_MADE_norm(self.before_trg_word_MADE(dec_output))
                # dec_output = self.before_trg_word_MADE(dec_output)
            if self.args.output_positional_embedding is not None:
                dec_output = torch.cat([outputPositionalEncoding(dec_output, self.args.output_positional_embedding), dec_output], dim=2)
            if self.args.feed_graph_length:
                dec_output = torch.cat([get_lengths(src_seq, self.args, self.binary_nums).float(), dec_output], dim=2)
            if self.separate_termination_bit:
                semifinal_enc_output = semifinal_enc_output.reshape(semifinal_enc_output.size(0),
                                                                    semifinal_enc_output.size(1),
                                                                    self.n_ensemble * self.d_model)

        if self.args.use_MADE:
            if self.args.separate_termination_bit:
                seq_logit = self.trg_word_MADE(torch.cat([dec_output, prepare_for_MADE(gold[:, :, 1:], self.args)], dim=2))
            else:
                seq_logit = self.trg_word_MADE(torch.cat([dec_output, prepare_for_MADE(gold, self.args)], dim=2))
        else:
            # seq_logit = self.trg_word_prj(dec_output)
            seq_logit = self.trg_word_prj_1(dec_output)
            seq_logit = self.trg_word_prj_2(nn.functional.relu(seq_logit))
            seq_logit = self.trg_word_prj_3(nn.functional.relu(seq_logit))

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        if self.separate_termination_bit:
            term_logit = self.termination_bit_prj_1(semifinal_enc_output)
            term_logit = self.termination_bit_prj_2(nn.functional.relu(term_logit))
            term_logit = self.termination_bit_prj_3(nn.functional.relu(term_logit))
            seq_logit = torch.cat([term_logit, seq_logit], dim=-1)

        return seq_logit.view(-1, seq_logit.size(2)), dec_output


# def ensembleMask(mask,n_ensemble):
#     ##  mask is sz_b * seq_len * seq_len
#     mask = mask.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, n_ensemble, 1, n_ensemble)
#     mask = mask.reshape(mask.size(0), mask.size(1) * mask.size(2), mask.size(3) * mask.size(4))
#     return mask
