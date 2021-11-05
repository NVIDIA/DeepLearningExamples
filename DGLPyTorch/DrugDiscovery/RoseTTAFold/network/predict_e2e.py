import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates
from RoseTTAFoldModel  import RoseTTAFoldModule_e2e
import util
from collections import namedtuple
from ffindex import *
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d
from trFold import TRFold

script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_module"     : 8,
        "n_module_str" : 4,
        "n_module_ref" : 4,
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
        "p_drop"       : 0.0,
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

REF_param = {
        "num_layers"    : 3,
        "num_channels"  : 32,
        "num_degrees"   : 3,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 32,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['REF_param'] = REF_param

# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]

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
        self.model = RoseTTAFoldModule_e2e(**MODEL_PARAM).to(self.device)

    def load_model(self, model_name, suffix='e2e'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True
    
    def predict(self, a3m_fn, out_prefix, hhr_fn=None, atab_fn=None, window=150, shift=75):
        msa = parse_a3m(a3m_fn)
        N, L = msa.shape
        #
        if hhr_fn != None:
            xyz_t, t1d, t0d = read_templates(L, ffdb, hhr_fn, atab_fn, n_templ=10)
        else:
            xyz_t = torch.full((1, L, 3, 3), np.nan).float()
            t1d = torch.zeros((1, L, 3)).float()
            t0d = torch.zeros((1,3)).float()
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
       
        could_load = self.load_model(self.model_name, suffix="e2e")
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()
        self.model.eval()
        with torch.no_grad():
            # do cropped prediction if protein is too big
            if L > window*2:
                prob_s = [np.zeros((L,L,NBIN[i]), dtype=np.float32) for  i in range(4)]
                count_1d = np.zeros((L,), dtype=np.float32)
                count_2d = np.zeros((L,L), dtype=np.float32)
                node_s = np.zeros((L,MODEL_PARAM['d_msa']), dtype=np.float32)
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
                        # Select template
                        input_t1d = t1d[:,:,sel].to(self.device) # (B, T, L, 3)
                        input_t2d = t2d[:,:,sel][:,:,:,sel].to(self.device)
                        #
                        print ("running crop: %d-%d/%d-%d"%(start_1, end_1, start_2, end_2), input_msa.shape)
                        with torch.cuda.amp.autocast():
                            logit_s, node, init_crds, pred_lddt = self.model(input_msa, input_seq, input_idx, t1d=input_t1d, t2d=input_t2d, return_raw=True)
                        #
                        # Not sure How can we merge init_crds.....
                        sub_idx = input_idx[0].cpu()
                        sub_idx_2d = np.ix_(sub_idx, sub_idx)
                        count_2d[sub_idx_2d] += 1.0
                        count_1d[sub_idx] += 1.0
                        node_s[sub_idx] += node[0].cpu().numpy()
                        for i_logit, logit in enumerate(logit_s):
                            prob = self.active_fn(logit.float()) # calculate distogram
                            prob = prob.squeeze(0).permute(1,2,0).cpu().numpy()
                            prob_s[i_logit][sub_idx_2d] += prob
                        del logit_s, node
                #
                # combine all crops
                for i in range(4):
                    prob_s[i] = prob_s[i] / count_2d[:,:,None]
                prob_in = np.concatenate(prob_s, axis=-1)
                node_s = node_s / count_1d[:, None]
                #
                # Do iterative refinement using SE(3)-Transformers
                # clear cache memory
                torch.cuda.empty_cache()
                #
                node_s = torch.tensor(node_s).to(self.device).unsqueeze(0)
                seq = msa[:,0].to(self.device)
                idx_pdb = idx_pdb.to(self.device)
                prob_in = torch.tensor(prob_in).to(self.device).unsqueeze(0)
                with torch.cuda.amp.autocast():
                    xyz, lddt = self.model(node_s, seq, idx_pdb, prob_s=prob_in, refine_only=True)
            else:
                msa = msa[:,:1000].to(self.device)
                seq = msa[:,0]
                idx_pdb = idx_pdb.to(self.device)
                t1d = t1d[:,:10].to(self.device)
                t2d = t2d[:,:10].to(self.device)
                with torch.cuda.amp.autocast():
                    logit_s, _, xyz, lddt = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d)
                prob_s = list()
                for logit in logit_s:
                    prob = self.active_fn(logit.float()) # distogram
                    prob = prob.reshape(-1, L, L).permute(1,2,0).cpu().numpy()
                    prob_s.append(prob)
        
        np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16), \
                            omega=prob_s[1].astype(np.float16),\
                            theta=prob_s[2].astype(np.float16),\
                            phi=prob_s[3].astype(np.float16))
        
        self.write_pdb(seq[0], xyz[0], idx_pdb[0], Bfacts=lddt[0], prefix="%s_init"%(out_prefix))
        
        # run TRFold
        prob_trF = list()
        for prob in prob_s:
            prob = torch.tensor(prob).permute(2,0,1).to(self.device)
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
            prob_trF.append(prob)
        xyz = xyz[0, :, 1]
        TRF = TRFold(prob_trF, fold_params)
        xyz = TRF.fold(xyz, batch=15, lr=0.1, nsteps=200)
        xyz = xyz.detach().cpu().numpy()
        # add O and Cb
        N = xyz[:,0,:]
        CA = xyz[:,1,:]
        C = xyz[:,2,:]
        O = self.extend(np.roll(N, -1, axis=0), CA, C, 1.231, 2.108, -3.142)
        xyz = np.concatenate((xyz, O[:,None,:]), axis=1)
        self.write_pdb(seq[0], xyz, idx_pdb[0], Bfacts=lddt[0], prefix=out_prefix)

    def extend(self, a,b,c, L,A,D):
        '''
        input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
        output: 4th coord
        '''
        N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
        bc = N(b-c)
        n = N(np.cross(b-a, bc))
        m = [bc,np.cross(n,bc),n]
        d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
        return c + sum([m*d for m,d in zip(m,d)])

    def write_pdb(self, seq, atoms, idx, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                            "A", idx[i]+1, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    ctr += 1

                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
                
                elif atoms.shape[1]==4:
                    for j,atm_j in enumerate((" N  "," CA "," C  ", " O  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
        

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_dir", default="%s/weights"%(script_dir),
                        help="Path to pre-trained network weights [%s/weights]"%script_dir)
    parser.add_argument("-i", dest="a3m_fn", required=True,
                        help="Input multiple sequence alignments (in a3m format)")
    parser.add_argument("-o", dest="out_prefix", required=True,
                        help="Prefix for output file. The output files will be [out_prefix].npz and [out_prefix].pdb")
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
