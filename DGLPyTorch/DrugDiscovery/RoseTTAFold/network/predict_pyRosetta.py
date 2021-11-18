import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d

script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_module"     : 8,
        "n_module_str" : 4,
        "n_layer"      : 1,
        "d_msa"        : 384 ,
        "d_pair"       : 288,
        "d_templ"      : 64,
        "n_head_msa"   : 12,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "r_ff"         : 4,
        "n_resblock"   : 1,
        "p_drop"       : 0.1,
        "use_templ"    : True,
        "performer_N_opts": {"nb_features": 64},
        "performer_L_opts": {"nb_features": 64}
        }

SE3_param = {
        "num_layers"    : 2,
        "num_channels"  : 16,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 32,
        "div": 2,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param

class Predictor():
    def __init__(self, model_dir=None, use_cpu=False):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.realpath(__file__)))
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "RoseTTAFold"
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name, suffix='pyrosetta'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return True
    
    def predict(self, a3m_fn, out_prefix, hhr_fn=None, atab_fn=None, window=150, shift=50):
        msa = parse_a3m(a3m_fn)
        N, L = msa.shape
        #
        if hhr_fn != None:
            xyz_t, t1d, t0d = read_templates(L, ffdb, hhr_fn, atab_fn, n_templ=25)
        else:
            xyz_t = torch.full((1, L, 3, 3), np.nan).float()
            t1d = torch.zeros((1, L, 3)).float()
            t0d = torch.zeros((1,3)).float()
       
        self.model.eval()
        with torch.no_grad():
            #
            msa = torch.tensor(msa).long().view(1, -1, L)
            idx_pdb = torch.arange(L).long().view(1, L)
            seq = msa[:,0]
            #
            # template features
            xyz_t = xyz_t.float().unsqueeze(0)
            t1d = t1d.float().unsqueeze(0)
            t0d = t0d.float().unsqueeze(0)
            t2d = xyz_to_t2d(xyz_t, t0d)
            #
            # do cropped prediction
            if L > window*2:
                prob_s = [np.zeros((L,L,NBIN[i]), dtype=np.float32) for  i in range(4)]
                count = np.zeros((L,L), dtype=np.float32)
                #
                grids = np.arange(0, L-window+shift, shift)
                ngrids = grids.shape[0]
                print("ngrid:     ", ngrids)
                print("grids:     ", grids)
                print("windows:   ", window)

                for i in range(ngrids):
                    for j in range(i, ngrids):
                        start_1 = grids[i]
                        end_1 = min(grids[i]+window, L)
                        start_2 = grids[j]
                        end_2 = min(grids[j]+window, L)
                        sel = np.zeros((L)).astype(np.bool)
                        sel[start_1:end_1] = True
                        sel[start_2:end_2] = True
                       
                        input_msa = msa[:,:,sel]
                        mask = torch.sum(input_msa==20, dim=-1) < 0.5*sel.sum() # remove too gappy sequences
                        input_msa = input_msa[mask].unsqueeze(0)
                        input_msa = input_msa[:,:1000].to(self.device)
                        input_idx = idx_pdb[:,sel].to(self.device)
                        input_seq = input_msa[:,0].to(self.device)
                        #
                        input_t1d = t1d[:,:,sel].to(self.device)
                        input_t2d = t2d[:,:,sel][:,:,:,sel].to(self.device)
                        #
                        print ("running crop: %d-%d/%d-%d"%(start_1, end_1, start_2, end_2), input_msa.shape)
                        with torch.cuda.amp.autocast():
                            logit_s, init_crds, pred_lddt = self.model(input_msa, input_seq, input_idx, t1d=input_t1d, t2d=input_t2d)
                        #
                        pred_lddt = torch.clamp(pred_lddt, 0.0, 1.0)
                        weight = pred_lddt[0][:,None] + pred_lddt[0][None,:]
                        weight = weight.cpu().numpy() + 1e-8
                        sub_idx = input_idx[0].cpu()
                        sub_idx_2d = np.ix_(sub_idx, sub_idx)
                        count[sub_idx_2d] += weight
                        for i_logit, logit in enumerate(logit_s):
                            prob = self.active_fn(logit.float()) # calculate distogram
                            prob = prob.squeeze(0).permute(1,2,0).cpu().numpy()
                            prob_s[i_logit][sub_idx_2d] += weight[:,:,None]*prob
                for i in range(4):
                    prob_s[i] = prob_s[i] / count[:,:,None]
            else:
                msa = msa[:,:1000].to(self.device)
                seq = msa[:,0]
                idx_pdb = idx_pdb.to(self.device)
                t1d = t1d.to(self.device)
                t2d = t2d.to(self.device)
                logit_s, init_crds, pred_lddt = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d)
                prob_s = list()
                for logit in logit_s:
                    prob = self.active_fn(logit.float()) # distogram
                    prob = prob.reshape(-1, L, L).permute(1,2,0).cpu().numpy()
                    prob_s.append(prob)
        
        np.savez_compressed("%s.npz"%out_prefix, dist=prob_s[0].astype(np.float16), \
                            omega=prob_s[1].astype(np.float16),\
                            theta=prob_s[2].astype(np.float16),\
                            phi=prob_s[3].astype(np.float16))
                    

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_dir", default="%s/weights"%(script_dir),
                        help="Path to pre-trained network weights [%s/weights]"%script_dir)
    parser.add_argument("-i", dest="a3m_fn", required=True,
                        help="Input multiple sequence alignments (in a3m format)")
    parser.add_argument("-o", dest="out_prefix", required=True,
                        help="Prefix for output file. The output file will be [out_prefix].npz")
    parser.add_argument("--hhr", default=None,
                        help="HHsearch output file (hhr file). If not provided, zero matrices will be given as templates")
    parser.add_argument("--atab", default=None,
                        help="HHsearch output file (atab file)")
    parser.add_argument("--db", default="%s/pdb100_2021Mar03/pdb100_2021Mar03"%script_dir,
                        help="Path to template database [%s/pdb100_2021Mar03]"%script_dir)
    parser.add_argument("--cpu", dest='use_cpu', default=False, action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    FFDB=args.db
    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                     read_data(FFDB+'_pdb.ffdata'))

    if not os.path.exists("%s.npz"%args.out_prefix):
        pred = Predictor(model_dir=args.model_dir, use_cpu=args.use_cpu)
        pred.predict(args.a3m_fn, args.out_prefix, args.hhr, args.atab)
