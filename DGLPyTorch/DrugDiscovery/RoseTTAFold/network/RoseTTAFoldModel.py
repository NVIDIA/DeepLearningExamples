import torch
import torch.nn as nn
from Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from Attention_module_w_str import IterativeFeatureExtractor
from DistancePredictor import DistanceNetwork
from Refine_module import Refine_module

class RoseTTAFoldModule(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(RoseTTAFoldModule, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)

    def forward(self, msa, seq, idx, t1d=None, t2d=None):
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)
        #
        # Extract features
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)

        # Predict 6D coords
        logits = self.c6d_predictor(pair)
        
        return logits, xyz, lddt.view(B, L)


class RoseTTAFoldModule_e2e(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_module_ref=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.0, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 REF_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(RoseTTAFoldModule_e2e, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)
        #
        self.refine = Refine_module(n_module_ref, d_node=d_msa, d_pair=130,
                                    d_node_hidden=d_hidden, d_pair_hidden=d_hidden,
                                    SE3_param=REF_param, p_drop=p_drop)

    def forward(self, msa, seq, idx, t1d=None, t2d=None, prob_s=None, return_raw=False, refine_only=False):
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        if not refine_only:
            B, N, L = msa.shape
            # Get embeddings
            msa = self.msa_emb(msa, idx)
            if self.use_templ:
                tmpl = self.templ_emb(t1d, t2d, idx)
                pair = self.pair_emb(seq, idx, tmpl)
            else:
                pair = self.pair_emb(seq, idx)
            #
            # Extract features
            msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)

            # Predict 6D coords
            logits = self.c6d_predictor(pair)
            
            prob_s = list()
            for l in logits:
                prob_s.append(nn.Softmax(dim=1)(l)) # (B, C, L, L)
            prob_s = torch.cat(prob_s, dim=1).permute(0,2,3,1)
        
        B, L = msa.shape[:2]
        if return_raw:
            return logits, msa, xyz, lddt.view(B, L)
        
        ref_xyz, ref_lddt = self.refine(msa, prob_s, seq1hot, idx)

        if refine_only:
            return ref_xyz, ref_lddt.view(B,L)
        else:
            return logits, msa, ref_xyz, ref_lddt.view(B,L)
