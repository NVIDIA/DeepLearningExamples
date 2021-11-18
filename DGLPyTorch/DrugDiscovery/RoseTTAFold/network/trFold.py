import numpy as np
import torch
import torch.nn as nn

def perturb_init(xyz, batch, noise=0.5):
    L = xyz.shape[0]
    pert = torch.tensor(np.random.uniform(noise, size=(batch, L, 3)), device=xyz.device)
    
    xyz = xyz.unsqueeze(0) + pert.detach()
    return xyz

def Q2R(Q):
    '''convert quaternions to rotation matrices'''
    b,l,_ = Q.shape
    w,x,y,z = Q[...,0],Q[...,1],Q[...,2],Q[...,3]
    xx,xy,xz,xw = x*x, x*y, x*z, x*w
    yy,yz,yw = y*y, y*z, y*w
    zz,zw = z*z, z*w
    R = torch.stack([1-2*yy-2*zz, 2*xy-2*zw,   2*xz+2*yw,
                     2*xy+2*zw,   1-2*xx-2*zz, 2*yz-2*xw,
                     2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy],dim=-1).view(b,l,3,3)
    return R

def get_cb(N,Ca,C):
    """recreate Cb given N,Ca,C"""
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    
    return Cb

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x)


class TRFold():

    def __init__(self, pred, params):

        self.pred = pred
        self.params = params
        self.device = self.pred[0].device
        
        # dfire background correction for distograms
        self.bkgd = (torch.linspace(4.25,19.75,32,device=self.device)/
                     self.params['DCUT'])**self.params['ALPHA']
        
        # background correction for phi
        ang = torch.linspace(0,np.pi,19,device=self.device)[:-1]
        self.bkgp = 0.5*(torch.cos(ang)-torch.cos(ang+np.deg2rad(10.0)))
        
        # Sav-Gol filter
        self.sg = torch.from_numpy(self.params['SG']).float().to(self.device)
        
        # paddings for distograms:
        # left - linear clash; right - zeroes
        padRsize = self.sg.shape[-1]//2+3
        padLsize = padRsize + 8
        padR = torch.zeros(padRsize,device=self.device)
        padL = torch.arange(1,padLsize+1,device=self.device).flip(0)*self.params['CLASH']
        self.padR = padR[:,None]
        self.padL = padL[:,None]
    
        # backbone motif
        self.ncac = torch.from_numpy(self.params['NCAC']).to(self.device)

    def akima(self, y,h):
        ''' Akima spline coefficients (boundaries trimmed to [2:-2])
        https://doi.org/10.1145/321607.321609 '''
        m = (y[:,1:]-y[:,:-1])/h
        #m += 1e-3*torch.randn(m.shape, device=m.device)
        m4m3 = torch.abs(m[:,3:]-m[:,2:-1])
        m2m1 = torch.abs(m[:,1:-2]-m[:,:-3])
        t = (m4m3*m[:,1:-2] + m2m1*m[:,2:-1])/(m4m3+m2m1)
        t[torch.isnan(t)] = 0.0
        dy = y[:,3:-2]-y[:,2:-3]
        coef = torch.stack([y[:,2:-3],
                            t[:,:-1],
                            (3*dy/h - 2*t[:,:-1] - t[:,1:])/h,
                            (t[:,:-1]+t[:,1:] - 2*dy/h)/h**2], dim=-1)
        return coef
        
    def fold(self, xyz, batch=32, lr=0.8, nsteps=100):

        pd,po,pt,pp = self.pred
        L = pd.shape[-1]

        p20 = (6.0-pd[-1]-po[-1]-pt[-1]-pp[-1]-(pt[-1]+pp[-1]).T)/6
        i,j = torch.triu_indices(L,L,1,device=self.device)
        sel = torch.where(p20[i,j]>self.params['PCUT'])[0]

        # indices for dist and omega (symmetric)
        i_s,j_s = i[sel], j[sel]
        
        # indices for theta and phi (asymmetric)
        i_a,j_a = torch.hstack([i_s,j_s]), torch.hstack([j_s,i_s]) 

        # background-corrected initial restraints
        cstd = -torch.log(pd[4:36,i_s,j_s]/self.bkgd[:,None])
        csto = -torch.log(po[0:36,i_s,j_s]/(1./36)) # omega and theta
        cstt = -torch.log(pt[0:36,i_a,j_a]/(1./36)) # are almost uniform
        cstp = -torch.log(pp[0:18,i_a,j_a]/self.bkgp[:,None])
        
        # padded restraints
        pad = self.sg.shape[-1]//2+3
        cstd = torch.cat([self.padL+cstd[0],cstd,self.padR+cstd[-1]],dim=0)
        csto = torch.cat([csto[-pad:],csto,csto[:pad]],dim=0)
        cstt = torch.cat([cstt[-pad:],cstt,cstt[:pad]],dim=0)
        cstp = torch.cat([cstp[:pad].flip(0),cstp,cstp[-pad:].flip(0)],dim=0)
        
        # smoothed restraints
        cstd,csto,cstt,cstp = [nn.functional.conv1d(cst.T.unsqueeze(1),self.sg)[:,0] 
                               for cst in [cstd,csto,cstt,cstp]]

        # force distance restraints vanish at long distances
        cstd = cstd-cstd[:,-1][:,None]

        # akima spline coefficients
        coefd = self.akima(cstd, self.params['DSTEP']).detach()
        coefo = self.akima(csto, self.params['ASTEP']).detach()
        coeft = self.akima(cstt, self.params['ASTEP']).detach()
        coefp = self.akima(cstp, self.params['ASTEP']).detach()

        astep = self.params['ASTEP']

        ko = torch.arange(i_s.shape[0],device=self.device).repeat(batch)
        kt = torch.arange(i_a.shape[0],device=self.device).repeat(batch)
        
        # initial Ca placement using EDM+minimization
        xyz = perturb_init(xyz, batch) # (batch, L, 3)
        
        # optimization variables: T - shift vectors, Q - rotation quaternions
        T = torch.zeros_like(xyz,device=self.device,requires_grad=True)
        Q = torch.randn([batch,L,4],device=self.device,requires_grad=True)
        bb0 = self.ncac[None,:,None,:].repeat(batch,1,L,1)
        
        opt = torch.optim.Adam([T,Q], lr=lr)
        for step in range(nsteps):


            R = Q2R(Q/torch.norm(Q,dim=-1,keepdim=True))
            bb = torch.einsum("blij,bklj->bkli",R,bb0)+(xyz+T)[:,None]

            # TODO: include Cb in the motif
            N,Ca,C = bb[:,0],bb[:,1],bb[:,2]
            Cb = get_cb(N,Ca,C)
            
            o = get_dih(Ca[:,i_s],Cb[:,i_s],Cb[:,j_s],Ca[:,j_s]) + np.pi
            t = get_dih(N[:,i_a],Ca[:,i_a],Cb[:,i_a],Cb[:,j_a]) + np.pi
            p = get_ang(Ca[:,i_a],Cb[:,i_a],Cb[:,j_a])
            
            dij = torch.norm(Cb[:,i_s]-Cb[:,j_s],dim=-1)
            b,k = torch.where(dij<20.0)
            dk = dij[b,k]

            #coord  = [coord/step-0.5 for coord,step in zip([dij,o,t,p],[dstep,astep,astep,astep])]
            #bins   = [torch.ceil(c).long() for c in coord]
            #delta  = [torch.frac(c) for c in coord]
            
            kbin = torch.ceil((dk-0.25)/0.5).long()
            dx = (dk-0.25)%0.5
            c = coefd[k,kbin]
            lossd = c[:,0]+c[:,1]*dx+c[:,2]*dx**2+c[:,3]*dx**3

            # omega
            obin = torch.ceil((o.view(-1)-astep/2)/astep).long()
            do = (o.view(-1)-astep/2)%astep
            co = coefo[ko,obin]
            losso = (co[:,0]+co[:,1]*do+co[:,2]*do**2+co[:,3]*do**3).view(batch,-1) #.mean(1)
            
            # theta
            tbin = torch.ceil((t.view(-1)-astep/2)/astep).long()
            dt = (t.view(-1)-astep/2)%astep
            ct = coeft[kt,tbin]
            losst = (ct[:,0]+ct[:,1]*dt+ct[:,2]*dt**2+ct[:,3]*dt**3).view(batch,-1) #.mean(1)

            # phi
            pbin = torch.ceil((p.view(-1)-astep/2)/astep).long()
            dp = (p.view(-1)-astep/2)%astep
            cp = coefp[kt,pbin]
            lossp = (cp[:,0]+cp[:,1]*dp+cp[:,2]*dp**2+cp[:,3]*dp**3).view(batch,-1) #.mean(1)
            
            # restrain geometry of peptide bonds
            loss_nc = (torch.norm(C[:,:-1]-N[:,1:],dim=-1)-1.32868)**2
            loss_cacn = (get_ang(Ca[:,:-1], C[:,:-1], N[:,1:]) - 2.02807)**2
            loss_canc = (get_ang(Ca[:,1:], N[:,1:], C[:,:-1]) - 2.12407)**2
            
            loss_geom = loss_nc.mean(1) + loss_cacn.mean(1) + loss_canc.mean(1)
            loss_ang = losst.mean(1) + losso.mean(1) + lossp.mean(1)
            
            # coefficient for ramping up geometric restraints during minimization
            coef = (1.0+step)/nsteps 
            
            loss = lossd.mean() + self.params['WANG']*loss_ang.mean() + coef*self.params['WCST']*loss_geom.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        lossd = torch.stack([lossd[b==i].mean() for i in range(batch)])
        loss = lossd + self.params['WANG']*loss_ang + self.params['WCST']*loss_geom
        minidx = torch.argmin(loss)
        
        return bb[minidx].permute(1,0,2)

