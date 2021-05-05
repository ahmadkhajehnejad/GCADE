''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from made.made import MADE


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
    return subsequent_mask


def outputPositionalEncoding(data):
    seq_len = data.size(1)
    # return torch.eye(seq_len, device=data.device).unsqueeze(0).repeat(data.size(0), 1, 1)
    return torch.tril( torch.ones(data.size(0), seq_len, seq_len, device=data.device), diagonal=0)


class PositionalEncoding(nn.Module):

    def __init__(self, args, d_hid, n_position=1000):
        super(PositionalEncoding, self).__init__()

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

    def forward(self, x):
        if self.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            return x + self.pos_table[:, :x.size(1)].unsqueeze(-2).clone().detach()
        elif self.input_type == 'node_based':
            return x + self.pos_table[:, :x.size(1)].clone().detach()
        else:
            raise NotImplementedError


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_ensemble, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, args, dropout=0.1, n_position=1000, scale_emb=False):

        super().__init__()

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
        self.position_enc = PositionalEncoding(args=args, d_hid=d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_ensemble, n_head, d_k, d_v, k_gr_att=args.k_graph_attention, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, gr_mask, return_attns=False):

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
        enc_output = self.dropout(self.position_enc(enc_output))
        # enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask, gr_mask=gr_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
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
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_ensemble, n_head, d_k, d_v, k_gr_att=args.k_graph_attention, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, gr_mask, return_attns=False):

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
        # dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask, gr_mask=gr_mask)
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

            sz_in = d_model * n_ensemble
            if args.output_positional_embedding:
                sz_in = sz_in + args.max_seq_len

            sz_intermed = max(sz_in, sz_out)


            if args.use_MADE:
                hidden_sizes = [sz_intermed * 3 // 2] * 3
                self.trg_word_MADE = MADE(sz_in, hidden_sizes, sz_out, num_masks=1, natural_ordering=True)
            else:
                # self.trg_word_prj = nn.Linear(sz_in, sz_out, bias=False)
                self.trg_word_prj_1 = nn.Linear(sz_in, sz_intermed, bias=True)
                self.trg_word_prj_2 = nn.Linear(sz_intermed, sz_intermed, bias=True)
                self.trg_word_prj_3 = nn.Linear(sz_intermed, sz_out, bias=True)
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


    def forward(self, src_seq, trg_seq, gold, adj):

        k_gr_att = self.args.k_graph_attention

        if k_gr_att > 0:
            gr_mask = torch.zeros(adj.size(0), k_gr_att, adj.size(1), adj.size(2)).to(self.args.device)
            gr_mask[:, 0, :, :] = torch.triu(adj)
            for i in range(1, k_gr_att):
                gr_mask[:, i, :, :] = torch.triu(torch.matmul(adj, gr_mask[:, i-1, :, :]))

            gr_mask = torch.transpose(gr_mask, 2, 3)
            if self.args.normalize_graph_attention:
                sm = gr_mask.sum(-1, keepdim=True)
                sm = sm.masked_fill(sm == 0, 1)
                gr_mask = gr_mask / sm

        else:
            gr_mask = None

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

        enc_output, *_ = self.encoder(src_seq, src_mask, gr_mask)
        if self.args.only_encoder:
            dec_output = enc_output
        else:
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask, gr_mask)

        if self.args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
            dec_output = dec_output.reshape(dec_output.size(0), dec_output.size(1), self.n_ensemble * self.d_model)
            if self.args.output_positional_embedding:
                dec_output = torch.cat([dec_output, outputPositionalEncoding(dec_output)], dim=2)

        if self.args.use_MADE:
            seq_logit = self.trg_word_MADE(torch.cat([dec_output, gold], dim=2))
        else:
            # seq_logit = self.trg_word_prj(dec_output)
            seq_logit = self.trg_word_prj_1(dec_output)
            seq_logit = self.trg_word_prj_2(nn.functional.relu(seq_logit))
            seq_logit = self.trg_word_prj_3(nn.functional.relu(seq_logit))

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2)), dec_output


# def ensembleMask(mask,n_ensemble):
#     ##  mask is sz_b * seq_len * seq_len
#     mask = mask.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, n_ensemble, 1, n_ensemble)
#     mask = mask.reshape(mask.size(0), mask.size(1) * mask.size(2), mask.size(3) * mask.size(4))
#     return mask
