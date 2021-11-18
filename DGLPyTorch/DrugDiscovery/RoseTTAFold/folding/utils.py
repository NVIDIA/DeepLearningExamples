import numpy as np
import random
import scipy
from scipy.signal import *
from pyrosetta import *

eps = 1e-9
P_ADD_OMEGA = 0.5
P_ADD_THETA = 0.5
P_ADD_PHI = 0.6

def gen_rst(params):

    npz = np.load(params['NPZ'])

    dist,omega,theta,phi = npz['dist'],npz['omega'],npz['theta'],npz['phi']

    if params['ROLL']==True:
        print("Apply circular shift...")
        dist  = np.roll(dist,1,axis=-1)
        omega = np.roll(omega,1,axis=-1)
        theta = np.roll(theta,1,axis=-1)
        phi   = np.roll(phi,1,axis=-1)

    dist = dist.astype(np.float32) + eps
    omega = omega.astype(np.float32) + eps
    theta = theta.astype(np.float32) + eps
    phi = phi.astype(np.float32) + eps

    # dictionary to store Rosetta restraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT   = 0.05 #params['PCUT']
    EBASE  = params['EBASE']
    EREP   = params['EREP']
    DREP   = params['DREP']
    PREP   = params['PREP']
    SIGD   = params['SIGD']
    SIGM   = params['SIGM']
    MEFF   = params['MEFF']
    DCUT   = params['DCUT']
    ALPHA  = params['ALPHA']
    BBWGHT = params['BBWGHT']

    DSTEP  = params['DSTEP']
    ASTEP  = np.deg2rad(params['ASTEP'])

    seq = params['seq']

    sg_flag = False
    if params['SG'] != '':
        sg_flag = True
        sg_w,sg_n = [int(v) for v in params['SG'].split(",")]
        print("Savitzky-Golay:     %d,%d"%(sg_w,sg_n))

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25+DSTEP*i for i in range(32)])
    prob = np.sum(dist[:,:,5:], axis=-1) # prob of dist within 20A
    prob_12 = np.sum(dist[:,:,5:21], axis=-1) # prob of dist within 12A
    bkgr = np.array((bins/DCUT)**ALPHA)
    attr = -np.log((dist[:,:,5:]+MEFF)/(dist[:,:,-1][:,:,None]*bkgr[None,None,:]))+EBASE
    repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
    dist = np.concatenate([repul,attr], axis=-1)
    bins = np.concatenate([DREP,bins])
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    #
    prob = np.triu(prob, k=1) # fill zeros to diagonal and lower (for speed-up)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    prob_12 = prob_12[i,j]
    #nbins = 35
    step = 0.5
    for a,b,p,p_12 in zip(i,j,prob,prob_12):
        y = pyrosetta.rosetta.utility.vector1_double()
        if sg_flag == True:
            _ = [y.append(v) for v in savgol_filter(dist[a,b],sg_w,sg_n)]
        else:
            _ = [y.append(v) for v in dist[a,b]]
        spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, step, x,y)
        ida = rosetta.core.id.AtomID(5,a+1)
        idb = rosetta.core.id.AtomID(5,b+1)
        rst['dist'].append([a,b,p,p_12,rosetta.core.scoring.constraints.AtomPairConstraint(ida, idb, spline)])
    print("dist restraints:    %d"%(len(rst['dist'])))


    ########################################################
    # omega: -pi..pi
    ########################################################
    nbins = omega.shape[2]-1
    ASTEP = 2.0*np.pi/nbins
    nbins += 4
    bins = np.linspace(-np.pi-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    prob = np.sum(omega[:,:,1:], axis=-1)
    prob = np.triu(prob, k=1) # fill zeros to diagonal and lower (for speed-up)
    i,j = np.where(prob>PCUT+P_ADD_OMEGA)
    prob = prob[i,j]
    omega = -np.log((omega+MEFF)/(omega[:,:,-1]+MEFF)[:,:,None])
    #if sg_flag == True:
    #    omega = savgol_filter(omega,sg_w,sg_n,axis=-1,mode='wrap')
    omega = np.concatenate([omega[:,:,-2:],omega[:,:,1:],omega[:,:,1:3]],axis=-1)
    for a,b,p in zip(i,j,prob):
        y = pyrosetta.rosetta.utility.vector1_double()
        _ = [y.append(v) for v in omega[a,b]]
        spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
        id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
        id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
        id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
        id4 = rosetta.core.id.AtomID(2,b+1) # CA-j
        rst['omega'].append([a,b,p,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
    print("omega restraints:   %d"%(len(rst['omega'])))


    ########################################################
    # theta: -pi..pi
    ########################################################
    prob = np.sum(theta[:,:,1:], axis=-1)
    np.fill_diagonal(prob, 0.0)
    i,j = np.where(prob>PCUT+P_ADD_THETA)
    prob = prob[i,j]
    theta = -np.log((theta+MEFF)/(theta[:,:,-1]+MEFF)[:,:,None])
    #if sg_flag == True:
    #    theta = savgol_filter(theta,sg_w,sg_n,axis=-1,mode='wrap')
    theta = np.concatenate([theta[:,:,-2:],theta[:,:,1:],theta[:,:,1:3]],axis=-1)
    for a,b,p in zip(i,j,prob):
        y = pyrosetta.rosetta.utility.vector1_double()
        _ = [y.append(v) for v in theta[a,b]]
        spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
        id1 = rosetta.core.id.AtomID(1,a+1) #  N-i
        id2 = rosetta.core.id.AtomID(2,a+1) # CA-i
        id3 = rosetta.core.id.AtomID(5,a+1) # CB-i
        id4 = rosetta.core.id.AtomID(5,b+1) # CB-j
        rst['theta'].append([a,b,p,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])

    print("theta restraints:   %d"%(len(rst['theta'])))


    ########################################################
    # phi: 0..pi
    ########################################################
    nbins = phi.shape[2]-1+4
    bins = np.linspace(-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    prob = np.sum(phi[:,:,1:], axis=-1)
    np.fill_diagonal(prob, 0.0)
    i,j = np.where(prob>PCUT+P_ADD_PHI)
    prob = prob[i,j]
    phi = -np.log((phi+MEFF)/(phi[:,:,-1]+MEFF)[:,:,None])
    #if sg_flag == True:
    #    phi = savgol_filter(phi,sg_w,sg_n,axis=-1,mode='mirror')
    phi = np.concatenate([np.flip(phi[:,:,1:3],axis=-1),phi[:,:,1:],np.flip(phi[:,:,-2:],axis=-1)], axis=-1)
    for a,b,p in zip(i,j,prob):
        y = pyrosetta.rosetta.utility.vector1_double()
        _ = [y.append(v) for v in phi[a,b]]
        spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
        id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
        id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
        id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
        rst['phi'].append([a,b,p,rosetta.core.scoring.constraints.AngleConstraint(id1,id2,id3, spline)])
    print("phi restraints:     %d"%(len(rst['phi'])))

    ########################################################
    # backbone torsions
    ########################################################
    if (params['BB'] != ''):
        bbnpz = np.load(params['BB'])
        bbphi,bbpsi = bbnpz['phi'],bbnpz['psi']
        rst['bbphi'] = []
        rst['bbpsi'] = []
        nbins = bbphi.shape[1]+4
        step = 2.*np.pi/bbphi.shape[1]
        bins = np.linspace(-1.5*step-np.pi, np.pi+1.5*step, nbins)
        x = pyrosetta.rosetta.utility.vector1_double()
        _ = [x.append(v) for v in bins]

        bbphi = -np.log(bbphi)
        bbphi = np.concatenate([bbphi[:,-2:],bbphi,bbphi[:,:2]],axis=-1).copy()

        bbpsi = -np.log(bbpsi)
        bbpsi = np.concatenate([bbpsi[:,-2:],bbpsi,bbpsi[:,:2]],axis=-1).copy()

        for i in range(1,nres):
            N1 = rosetta.core.id.AtomID(1,i)
            Ca1 = rosetta.core.id.AtomID(2,i)
            C1 = rosetta.core.id.AtomID(3,i)
            N2 = rosetta.core.id.AtomID(1,i+1)
            Ca2 = rosetta.core.id.AtomID(2,i+1)
            C2 = rosetta.core.id.AtomID(3,i+1)

            # psi(i)
            ypsi = pyrosetta.rosetta.utility.vector1_double()
            _ = [ypsi.append(v) for v in bbpsi[i-1]]
            spsi = rosetta.core.scoring.func.SplineFunc("", BBWGHT, 0.0, step, x,ypsi)
            rst['bbpsi'].append(rosetta.core.scoring.constraints.DihedralConstraint(N1,Ca1,C1,N2, spsi))

            # phi(i+1)
            yphi = pyrosetta.rosetta.utility.vector1_double()
            _ = [yphi.append(v) for v in bbphi[i]]
            sphi = rosetta.core.scoring.func.SplineFunc("", BBWGHT, 0.0, step, x,yphi)
            rst['bbphi'].append(rosetta.core.scoring.constraints.DihedralConstraint(C1,N2,Ca2,C2, sphi))

        print("bbbtor restraints:  %d"%(len(rst['bbphi'])+len(rst['bbpsi'])))

    return rst

def set_predicted_dihedral(pose, phi, psi, omega):

    nbins = phi.shape[1]
    bins = np.linspace(-180.,180.,nbins+1)[:-1] + 180./nbins

    nres = pose.total_residue()
    for i in range(nres):
        pose.set_phi(i+1,np.random.choice(bins,p=phi[i]))
        pose.set_psi(i+1,np.random.choice(bins,p=psi[i]))

        if np.random.uniform() < omega[i,0]:
            pose.set_omega(i+1,0)
        else:
            pose.set_omega(i+1,180)

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres+1):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)


#pick phi/psi randomly from:
#-140  153 180 0.135 B
# -72  145 180 0.155 B
#-122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)


def read_fasta(file):
    fasta=""
    first = True
    with open(file, "r") as f:
        for line in f:
            if(line[0] == ">"):
                if first:
                    first = False
                    continue
                else:
                    break
            else:
                line=line.rstrip()
                fasta = fasta + line;
    return fasta


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst, sep1, sep2, params, nogly=False, use_orient=None, pcut=None, p12_cut=0.0):
    if use_orient == None:
        use_orient = params['USE_ORIENT']
    if pcut == None:
        pcut=params['PCUT']
    
    seq = params['seq']

    # collect restraints
    array = []

    if nogly==True:
        dist_r = [r for a,b,p,p_12,r in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut and p_12>=p12_cut]
        if use_orient:
            omega_r = [r for a,b,p,r in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+P_ADD_OMEGA] #0.5
            theta_r = [r for a,b,p,r in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+P_ADD_THETA] #0.5
            phi_r   = [r for a,b,p,r in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+P_ADD_PHI] #0.6
    else:
        dist_r = [r for a,b,p,p_12,r in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut and p_12>=p12_cut]
        if use_orient:
            omega_r = [r for a,b,p,r in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+P_ADD_OMEGA]
            theta_r = [r for a,b,p,r in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+P_ADD_THETA]
            phi_r   = [r for a,b,p,r in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+P_ADD_PHI] #0.6

    #if params['BB'] != '':
    #    array += [r for r in rst['bbphi']]
    #    array += [r for r in rst['bbpsi']]
    array += dist_r
    if use_orient:
        array += omega_r
        array += theta_r
        array += phi_r

    if len(array) < 1:
        return

    print ("Number of applied pair restraints:", len(array))
    print (" - Distance restraints:", len(dist_r))
    if use_orient:
        print (" - Omega restraints:", len(omega_r))
        print (" - Theta restraints:", len(theta_r))
        print (" - Phi restraints:  ", len(phi_r))

    #random.shuffle(array)

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in array]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)

def add_crd_rst(pose, nres, std=1.0, tol=1.0):
    flat_har = rosetta.core.scoring.func.FlatHarmonicFunc(0.0, std, tol)
    rst = list()
    for i in range(1, nres+1):
        xyz = pose.residue(i).atom("CA").xyz() # xyz coord of CA atom
        ida = rosetta.core.id.AtomID(2,i) # CA idx for residue i
        rst.append(rosetta.core.scoring.constraints.CoordinateConstraint(ida, ida, xyz, flat_har)) 

    if len(rst) < 1:
        return
    
    print ("Number of applied coordinate restraints:", len(rst))
    #random.shuffle(rst)

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in rst]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)

